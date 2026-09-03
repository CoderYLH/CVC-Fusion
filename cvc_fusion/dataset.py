"""Paired one-dimensional signal and spectrogram-image dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


def _resize_and_center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = size / min(width, height)
    resized_width = max(size, int(round(width * scale)))
    resized_height = max(size, int(round(height * scale)))
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    image = image.resize((resized_width, resized_height), resampling)
    left = (resized_width - size) // 2
    top = (resized_height - size) // 2
    return image.crop((left, top, left + size, top + size))


def _image_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(array)


class PairedVocalizationDataset(Dataset):
    """Load matched XLSX signals and PNG spectrograms from class directories."""

    def __init__(
        self,
        audio_root: str | Path,
        image_root: str | Path,
        class_names: Sequence[str],
        image_size: int = 224,
        audio_length: int | None = 100,
    ) -> None:
        self.audio_root = Path(audio_root)
        self.image_root = Path(image_root)
        self.class_names = [str(name) for name in class_names]
        self.image_size = image_size
        self.audio_length = audio_length
        self.samples: list[tuple[Path, Path, int]] = []

        if not self.audio_root.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_root}")
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")

        for label, class_name in enumerate(self.class_names):
            audio_directory = self.audio_root / class_name
            image_directory = self.image_root / class_name
            if not audio_directory.is_dir() or not image_directory.is_dir():
                raise FileNotFoundError(
                    f"Expected class directory '{class_name}' under both roots"
                )

            audio_files = {
                path.stem: path
                for path in audio_directory.glob("*.xlsx")
                if not path.name.startswith("~$")
            }
            image_files = {path.stem: path for path in image_directory.glob("*.png")}
            missing_images = sorted(set(audio_files) - set(image_files))
            missing_audio = sorted(set(image_files) - set(audio_files))
            if missing_images or missing_audio:
                raise ValueError(
                    f"Unpaired files in class '{class_name}': "
                    f"{len(missing_images)} signals without images and "
                    f"{len(missing_audio)} images without signals"
                )
            self.samples.extend(
                (audio_files[stem], image_files[stem], label)
                for stem in sorted(audio_files)
            )

        if not self.samples:
            raise ValueError("No paired XLSX/PNG samples were found")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        audio_path, image_path, label = self.samples[index]
        audio = pd.read_excel(audio_path, header=None).iloc[:, 0].to_numpy(
            dtype=np.float32
        )
        if self.audio_length is not None and audio.shape[0] != self.audio_length:
            raise ValueError(
                f"Expected {self.audio_length} values in {audio_path}, "
                f"but found {audio.shape[0]}"
            )
        audio_tensor = torch.from_numpy(audio)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = _resize_and_center_crop(image, self.image_size)
            image_tensor = _image_to_tensor(image)

        return {
            "audio": audio_tensor,
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "filename": image_path.name,
        }


def collate_paired_samples(
    batch: list[dict[str, Tensor | str]],
) -> dict[str, Tensor | list[str]]:
    """Pad variable-length one-dimensional signals with zeros."""
    max_length = max(int(sample["audio"].shape[0]) for sample in batch)
    audio = torch.zeros((len(batch), max_length), dtype=torch.float32)
    images = torch.stack([sample["image"] for sample in batch])
    labels = torch.stack([sample["label"] for sample in batch])
    filenames: list[str] = []

    for index, sample in enumerate(batch):
        signal = sample["audio"]
        audio[index, : signal.shape[0]] = signal
        filenames.append(str(sample["filename"]))

    return {
        "audio": audio,
        "image": images,
        "label": labels,
        "filename": filenames,
    }
