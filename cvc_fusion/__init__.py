"""Minimal training implementation of the CVC-Fusion model."""

from .dataset import PairedVocalizationDataset, collate_paired_samples
from .model import CVCFusion

__all__ = ["CVCFusion", "PairedVocalizationDataset", "collate_paired_samples"]
