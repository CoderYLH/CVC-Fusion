"""CVC-Fusion model components.

The image encoder is adapted from the MobileViT/CVNets implementation. The
implementation retains only the modules used by CVC-Fusion training.
"""

from __future__ import annotations

import math
from copy import deepcopy
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round a channel count while avoiding a reduction greater than 10%."""
    min_value = divisor if min_value is None else min_value
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class ConvNormAct2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        use_norm: bool = True,
        use_act: bool = True,
    ) -> None:
        padding = ((kernel_size - 1) // 2) * dilation
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.1))
        if use_act:
            layers.append(nn.SiLU(inplace=False))
        super().__init__(*layers)


class ConvNormAct1d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        use_norm: bool = True,
        use_act: bool = True,
    ) -> None:
        padding = ((kernel_size - 1) // 2) * dilation
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if use_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        if use_act:
            layers.append(nn.ReLU())
        super().__init__(*layers)


class InvertedResidual2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        hidden_dim = make_divisible(round(in_channels * expand_ratio), 8)
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvNormAct2d(in_channels, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvNormAct2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    dilation=dilation,
                    groups=hidden_dim,
                ),
                ConvNormAct2d(
                    hidden_dim,
                    out_channels,
                    kernel_size=1,
                    use_act=False,
                ),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        output = self.block(x)
        return x + output if self.use_residual else output


class InvertedResidual1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        hidden_dim = make_divisible(round(in_channels * expand_ratio), 8)
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvNormAct1d(in_channels, hidden_dim, kernel_size=1))

        depthwise_groups = (
            hidden_dim
            if in_channels % hidden_dim == 0 and out_channels % hidden_dim == 0
            else 1
        )
        layers.extend(
            [
                ConvNormAct1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    groups=depthwise_groups,
                ),
                ConvNormAct1d(
                    hidden_dim,
                    out_channels,
                    kernel_size=1,
                    use_act=False,
                ),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        output = self.block(x)
        return x + output if self.use_residual else output


class LinearSelfAttention(nn.Module):
    """Separable self-attention with linear complexity."""

    def __init__(self, embed_dim: int, attention_dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_projection = nn.Conv2d(
            embed_dim, 1 + 2 * embed_dim, kernel_size=1, bias=True
        )
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_projection = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=1, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        query, key, value = torch.split(
            self.qkv_projection(x), [1, self.embed_dim, self.embed_dim], dim=1
        )
        context_scores = self.attention_dropout(F.softmax(query, dim=-1))
        context_vector = torch.sum(key * context_scores, dim=-1, keepdim=True)
        output = F.relu(value) * context_vector.expand_as(value)
        return self.output_projection(output)


class HFSBlock(nn.Module):
    """Hybrid Feature Selection block used in the image encoder."""

    def __init__(
        self,
        in_channels: int,
        attention_dim: int,
        number_attention_blocks: int,
        patch_size: tuple[int, int] = (2, 2),
        dropout: float = 0.0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.patch_height, self.patch_width = patch_size
        self.local_representation = nn.Sequential(
            ConvNormAct2d(
                in_channels,
                in_channels,
                kernel_size=3,
                groups=in_channels,
                dilation=dilation,
            ),
            ConvNormAct2d(
                in_channels,
                attention_dim,
                kernel_size=1,
                use_norm=False,
                use_act=False,
            ),
        )
        self.global_representation = nn.Sequential(
            *[
                LinearSelfAttention(attention_dim, attention_dropout=dropout)
                for _ in range(number_attention_blocks)
            ],
            nn.GroupNorm(1, attention_dim),
        )
        self.feature_weight = nn.Linear(2 * attention_dim, 2)
        self.global_average_pool = nn.AdaptiveAvgPool2d(1)
        self.output_projection = ConvNormAct2d(
            2 * attention_dim,
            in_channels,
            kernel_size=1,
            use_act=False,
        )

    def _resize_if_needed(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        if height % self.patch_height == 0 and width % self.patch_width == 0:
            return x
        new_height = math.ceil(height / self.patch_height) * self.patch_height
        new_width = math.ceil(width / self.patch_width) * self.patch_width
        return F.interpolate(
            x, size=(new_height, new_width), mode="bilinear", align_corners=True
        )

    def _unfold(self, feature_map: Tensor) -> tuple[Tensor, tuple[int, int]]:
        batch_size, channels, height, width = feature_map.shape
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )
        patches = patches.reshape(
            batch_size,
            channels,
            self.patch_height * self.patch_width,
            -1,
        )
        return patches, (height, width)

    def _fold(self, patches: Tensor, output_size: tuple[int, int]) -> Tensor:
        batch_size, channels, patch_area, number_patches = patches.shape
        patches = patches.reshape(batch_size, channels * patch_area, number_patches)
        return F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._resize_if_needed(x)
        local_features = self.local_representation(x)
        patches, output_size = self._unfold(local_features)
        patches = self.global_representation(patches)
        global_features = self._fold(patches, output_size)

        pooled_local = self.global_average_pool(local_features).flatten(1)
        pooled_global = self.global_average_pool(global_features).flatten(1)
        weights = F.softmax(
            self.feature_weight(torch.cat((pooled_local, pooled_global), dim=1)),
            dim=1,
        )
        local_weight = weights[:, 0].view(-1, 1, 1, 1)
        global_weight = weights[:, 1].view(-1, 1, 1, 1)
        fused_features = torch.cat(
            (local_weight * local_features, global_weight * global_features), dim=1
        )
        return self.output_projection(fused_features) + x


class MobileNetV2OneDimensional(nn.Module):
    """One-dimensional MobileNetV2 feature encoder."""

    def __init__(self, number_classes: int = 2, width_multiplier: float = 1.0) -> None:
        super().__init__()
        layer_configuration = (
            (1, 16, 1, 1),
            (2, 32, 1, 2),
        )
        self.stem = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        in_channels = 16
        stages: list[nn.Module] = []
        for expansion, channels, number_blocks, stride in layer_configuration:
            out_channels = make_divisible(channels * width_multiplier, 8)
            blocks: list[nn.Module] = []
            for block_index in range(number_blocks):
                blocks.append(
                    InvertedResidual1d(
                        in_channels,
                        out_channels,
                        stride=stride if block_index == 0 else 1,
                        expand_ratio=expansion,
                    )
                )
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.expansion = ConvNormAct1d(in_channels, 64, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, number_classes),
        )
        self.feature_dimension = 64
        self._initialize_non_convolutional_layers()

    def _initialize_non_convolutional_layers(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.stem(x.unsqueeze(1))
        x = self.stages(x)
        x = self.expansion(x)
        return self.classifier[0](x)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.expansion(self.stages(self.stem(x.unsqueeze(1)))))


class MobileViTImageEncoder(nn.Module):
    """MobileViTv3 image encoder with HFS blocks."""

    def __init__(self, number_classes: int = 2, width_multiplier: float = 0.5) -> None:
        super().__init__()
        stem_channels = make_divisible(
            32 * width_multiplier, divisor=8, min_value=16
        )
        self.stem = ConvNormAct2d(3, stem_channels, kernel_size=3, stride=2)

        layer_configuration = (
            {
                "out_channels": make_divisible(64 * width_multiplier, 16),
                "number_blocks": 1,
                "stride": 1,
                "block_type": "mobilenet",
            },
            {
                "out_channels": make_divisible(128 * width_multiplier, 8),
                "number_blocks": 2,
                "stride": 2,
                "block_type": "mobilenet",
            },
            {
                "out_channels": make_divisible(256 * width_multiplier, 8),
                "attention_dim": make_divisible(128 * width_multiplier, 8),
                "attention_blocks": 2,
                "stride": 2,
                "block_type": "mobilevit",
            },
            {
                "out_channels": make_divisible(384 * width_multiplier, 8),
                "attention_dim": make_divisible(192 * width_multiplier, 8),
                "attention_blocks": 4,
                "stride": 2,
                "block_type": "mobilevit",
            },
            {
                "out_channels": make_divisible(512 * width_multiplier, 8),
                "attention_dim": make_divisible(256 * width_multiplier, 8),
                "attention_blocks": 3,
                "stride": 2,
                "block_type": "mobilevit",
            },
        )

        in_channels = stem_channels
        stages: list[nn.Module] = []
        for configuration in layer_configuration:
            out_channels = int(configuration["out_channels"])
            blocks: list[nn.Module] = []
            if configuration["block_type"] == "mobilenet":
                for block_index in range(int(configuration["number_blocks"])):
                    blocks.append(
                        InvertedResidual2d(
                            in_channels,
                            out_channels,
                            stride=(
                                int(configuration["stride"])
                                if block_index == 0
                                else 1
                            ),
                            expand_ratio=2,
                        )
                    )
                    in_channels = out_channels
            else:
                blocks.append(
                    InvertedResidual2d(
                        in_channels,
                        out_channels,
                        stride=int(configuration["stride"]),
                        expand_ratio=2,
                    )
                )
                in_channels = out_channels
                blocks.append(
                    HFSBlock(
                        in_channels=in_channels,
                        attention_dim=int(configuration["attention_dim"]),
                        number_attention_blocks=int(
                            configuration["attention_blocks"]
                        ),
                        patch_size=(2, 2),
                    )
                )
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.feature_dimension = in_channels
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, number_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.stages(self.stem(x))
        return self.classifier[0](x).flatten(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.stages(self.stem(x)))


class CVCFusion(nn.Module):
    """Dual-encoder CVC-Fusion classifier."""

    def __init__(
        self,
        number_classes: int = 2,
        image_width_multiplier: float = 0.5,
        fusion_dimension: int = 256,
    ) -> None:
        super().__init__()
        self.audio_encoder = MobileNetV2OneDimensional(
            number_classes=number_classes
        )
        self.image_encoder = MobileViTImageEncoder(
            number_classes=number_classes,
            width_multiplier=image_width_multiplier,
        )
        self.audio_projection = nn.Linear(
            self.audio_encoder.feature_dimension, fusion_dimension
        )
        self.image_projection = nn.Linear(
            self.image_encoder.feature_dimension, fusion_dimension
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fusion_dimension, number_classes, bias=True),
        )

    def forward(self, audio: Tensor, image: Tensor) -> Tensor:
        audio_features = self.audio_encoder.extract_features(audio).squeeze(-1)
        image_features = self.image_encoder.extract_features(image)
        fused_features = self.audio_projection(audio_features) + self.image_projection(
            image_features
        )
        return self.classifier(fused_features)


class ExponentialMovingAverage:
    """Maintain an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, momentum: float = 0.0005) -> None:
        self.model = deepcopy(model).eval()
        self.momentum = momentum
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for name, averaged_value in self.model.state_dict().items():
            current_value = model_state[name].detach()
            if torch.is_floating_point(averaged_value):
                averaged_value.mul_(1.0 - self.momentum).add_(
                    current_value, alpha=self.momentum
                )
            else:
                averaged_value.copy_(current_value)

    def state_dict(self) -> dict[str, Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        self.model.load_state_dict(state_dict)
