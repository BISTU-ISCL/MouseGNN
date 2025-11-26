"""Dataset utilities for ST-GCN on mouse skeleton sequences."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from mousegnn.data.behaviors import BEHAVIORS, behavior_to_index


@dataclass
class SequenceItem:
    """A single skeleton sequence and metadata."""

    keypoints: np.ndarray  # shape (T, V, C)
    label: str
    video: Optional[str] = None


class SkeletonSequenceDataset(Dataset):
    """Dataset that loads ``.npz`` skeleton clips.

    Each ``.npz`` file must contain:
        * ``keypoints``: ``(T, V, C)`` array of x/y[/confidence] coordinates.
        * ``label``: string label name matching :data:`BEHAVIORS`.
    Optional fields:
        * ``video``: source video identifier for debugging.
    """

    def __init__(self, root: Path, transform: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.root}")
        self.transform = transform
        self.label_map = behavior_to_index()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        npz = np.load(self.files[idx], allow_pickle=True)
        keypoints = npz["keypoints"].astype(np.float32)  # (T, V, C)
        label_name = str(npz["label"])
        if label_name not in self.label_map:
            raise ValueError(f"Label {label_name} not in known behaviors {list(self.label_map)}")

        if self.transform:
            keypoints = self.transform(keypoints)

        # reshape to (C, T, V)
        keypoints = np.transpose(keypoints, (2, 0, 1))
        tensor = torch.from_numpy(keypoints)
        return tensor, self.label_map[label_name]


class NormalizeByBody(torch.nn.Module):
    """Normalize skeleton coordinates to zero mean and unit variance."""

    def __init__(self):
        super().__init__()

    def forward(self, keypoints: np.ndarray) -> np.ndarray:
        mean = keypoints.mean(axis=(0, 1), keepdims=True)
        std = keypoints.std(axis=(0, 1), keepdims=True) + 1e-6
        return (keypoints - mean) / std


class ClipSampler:
    """Sample fixed-length clips from sequences."""

    def __init__(self, clip_len: int, stride: int = 1):
        self.clip_len = clip_len
        self.stride = stride

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        T = sequence.shape[0]
        if T == self.clip_len:
            return sequence
        if T < self.clip_len:
            pad = self.clip_len - T
            last = np.repeat(sequence[-1:], pad, axis=0)
            return np.concatenate([sequence, last], axis=0)
        start = np.random.randint(0, max(1, T - self.clip_len + 1), dtype=int)
        return sequence[start : start + self.clip_len : self.stride]


class Compose:
    def __init__(self, transforms: Sequence[Callable[[np.ndarray], np.ndarray]]):
        self.transforms = transforms

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            arr = t(arr)
        return arr
