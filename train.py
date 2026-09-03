"""Train CVC-Fusion on pre-partitioned paired vocalization data."""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD
from torch.utils.data import DataLoader

from cvc_fusion import CVCFusion, PairedVocalizationDataset, collate_paired_samples
from cvc_fusion.model import ExponentialMovingAverage


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(device_setting: str) -> torch.device:
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_setting)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
    return device


def make_loader(
    dataset: PairedVocalizationDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    pin_memory: bool,
    prefetch_factor: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    arguments = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_paired_samples,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": False,
    }
    if workers > 0:
        arguments["prefetch_factor"] = prefetch_factor
        arguments["persistent_workers"] = False
    return DataLoader(**arguments)


class WarmupCosineSchedule:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        maximum_lr: float,
        minimum_lr: float,
        epochs: int,
        warmup_iterations: int,
        warmup_initial_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.maximum_lr = maximum_lr
        self.minimum_lr = minimum_lr
        self.epochs = epochs
        self.warmup_iterations = warmup_iterations
        self.warmup_initial_lr = warmup_initial_lr

    def step(self, epoch: int, global_step: int) -> float:
        if global_step < self.warmup_iterations:
            fraction = global_step / max(1, self.warmup_iterations)
            learning_rate = self.warmup_initial_lr + fraction * (
                self.maximum_lr - self.warmup_initial_lr
            )
        else:
            learning_rate = self.minimum_lr + 0.5 * (
                self.maximum_lr - self.minimum_lr
            ) * (1.0 + math.cos(math.pi * epoch / self.epochs))
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        return learning_rate


def move_batch(
    batch: dict[str, Tensor | list[str]],
    device: torch.device,
    channels_last: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    audio = batch["audio"].to(device, non_blocking=True)
    image = batch["image"].to(device, non_blocking=True)
    if channels_last:
        image = image.contiguous(memory_format=torch.channels_last)
    labels = batch["label"].to(device, non_blocking=True)
    return audio, image, labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    schedule: WarmupCosineSchedule,
    ema: ExponentialMovingAverage | None,
    device: torch.device,
    epoch: int,
    global_step: int,
    mixed_precision: bool,
    channels_last: bool,
    gradient_clip: float | None,
    log_frequency: int,
) -> tuple[float, float, int, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    learning_rate = optimizer.param_groups[0]["lr"]

    for batch_index, batch in enumerate(loader):
        learning_rate = schedule.step(epoch, global_step)
        audio, image, labels = move_batch(batch, device, channels_last)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=mixed_precision):
            logits = model(audio, image)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if gradient_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += batch_size
        global_step += 1

        if batch_index % log_frequency == 0:
            print(
                f"epoch={epoch + 1} batch={batch_index + 1}/{len(loader)} "
                f"loss={loss.item():.5f} lr={learning_rate:.8f}"
            )

    return running_loss / total, correct / total, global_step, learning_rate


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool,
    channels_last: bool,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        audio, image, labels = move_batch(batch, device, channels_last)
        with autocast(enabled=mixed_precision):
            logits = model(audio, image)
            loss = criterion(logits, labels)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += batch_size
    return running_loss / total, correct / total


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ExponentialMovingAverage | None,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_accuracy: float,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_accuracy": best_accuracy,
    }
    if ema is not None:
        state["ema_model"] = ema.state_dict()
    torch.save(state, path)


def append_metrics(path: Path, row: dict[str, float | int]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    arguments = parse_arguments()
    with arguments.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = select_device(str(config["training"]["device"]))
    use_cuda = device.type == "cuda"
    channels_last = bool(config["training"]["channels_last"]) and use_cuda
    mixed_precision = bool(config["training"]["mixed_precision"]) and use_cuda

    data_config = config["data"]
    class_names = [str(name) for name in data_config["class_names"]]
    image_size = int(data_config["image_size"])
    audio_length = int(data_config["audio_length"])
    train_dataset = PairedVocalizationDataset(
        data_config["train_audio_root"],
        data_config["train_image_root"],
        class_names,
        image_size,
        audio_length,
    )
    validation_dataset = PairedVocalizationDataset(
        data_config["validation_audio_root"],
        data_config["validation_image_root"],
        class_names,
        image_size,
        audio_length,
    )

    loader_config = config["loader"]
    train_loader = make_loader(
        train_dataset,
        batch_size=int(loader_config["train_batch_size"]),
        workers=int(loader_config["workers"]),
        shuffle=True,
        pin_memory=bool(loader_config["pin_memory"]) and use_cuda,
        prefetch_factor=int(loader_config["prefetch_factor"]),
        seed=seed,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=int(loader_config["validation_batch_size"]),
        workers=int(loader_config["workers"]),
        shuffle=False,
        pin_memory=bool(loader_config["pin_memory"]) and use_cuda,
        prefetch_factor=int(loader_config["prefetch_factor"]),
        seed=seed,
    )

    model_config = config["model"]
    model = CVCFusion(
        number_classes=len(class_names),
        image_width_multiplier=float(model_config["image_width_multiplier"]),
        fusion_dimension=int(model_config["fusion_dimension"]),
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    training_config = config["training"]
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training_config["label_smoothing"])
    )
    optimizer = SGD(
        model.parameters(),
        lr=float(training_config["maximum_learning_rate"]),
        momentum=float(training_config["momentum"]),
        weight_decay=float(training_config["weight_decay"]),
        nesterov=bool(training_config["nesterov"]),
    )
    schedule = WarmupCosineSchedule(
        optimizer,
        maximum_lr=float(training_config["maximum_learning_rate"]),
        minimum_lr=float(training_config["minimum_learning_rate"]),
        epochs=int(training_config["epochs"]),
        warmup_iterations=int(training_config["warmup_iterations"]),
        warmup_initial_lr=float(training_config["warmup_initial_learning_rate"]),
    )
    scaler = GradScaler(enabled=mixed_precision)
    ema = (
        ExponentialMovingAverage(model, float(training_config["ema_momentum"]))
        if bool(training_config["ema_enabled"])
        else None
    )

    output_directory = Path(training_config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    copied_config = output_directory / "config.yaml"
    if arguments.config.resolve() != copied_config.resolve():
        shutil.copyfile(arguments.config, copied_config)

    start_epoch = 0
    global_step = 0
    best_accuracy = -math.inf
    if arguments.resume is not None:
        checkpoint = torch.load(arguments.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if ema is not None and "ema_model" in checkpoint:
            ema.load_state_dict(checkpoint["ema_model"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_accuracy = float(checkpoint["best_accuracy"])

    print(
        f"device={device} train_samples={len(train_dataset)} "
        f"validation_samples={len(validation_dataset)} "
        f"parameters={sum(parameter.numel() for parameter in model.parameters())}"
    )

    epochs = int(training_config["epochs"])
    gradient_clip = training_config.get("gradient_clip")
    gradient_clip = None if gradient_clip is None else float(gradient_clip)
    for epoch in range(start_epoch, epochs):
        train_loss, train_accuracy, global_step, learning_rate = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            schedule,
            ema,
            device,
            epoch,
            global_step,
            mixed_precision,
            channels_last,
            gradient_clip,
            int(training_config["log_frequency"]),
        )
        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
            device,
            mixed_precision,
            channels_last,
        )
        ema_validation_loss = math.nan
        ema_validation_accuracy = math.nan
        if ema is not None:
            ema_validation_loss, ema_validation_accuracy = validate(
                ema.model,
                validation_loader,
                criterion,
                device,
                mixed_precision,
                channels_last,
            )

        is_best = validation_accuracy >= best_accuracy
        best_accuracy = max(best_accuracy, validation_accuracy)
        save_checkpoint(
            output_directory / "checkpoint_last.pt",
            model,
            ema,
            optimizer,
            scaler,
            epoch,
            global_step,
            best_accuracy,
        )
        if is_best:
            save_checkpoint(
                output_directory / "checkpoint_best.pt",
                model,
                ema,
                optimizer,
                scaler,
                epoch,
                global_step,
                best_accuracy,
            )

        metrics = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "ema_validation_loss": ema_validation_loss,
            "ema_validation_accuracy": ema_validation_accuracy,
        }
        append_metrics(output_directory / "metrics.csv", metrics)
        print(
            f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.5f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"validation_loss={validation_loss:.5f} "
            f"validation_accuracy={validation_accuracy:.4f} "
            f"ema_validation_accuracy={ema_validation_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
