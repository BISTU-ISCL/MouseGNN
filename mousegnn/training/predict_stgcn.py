"""Run inference with a trained ST-GCN checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml

from mousegnn.data.behaviors import BEHAVIORS
from mousegnn.datasets.skeleton_sequences import Compose, NormalizeByBody
from mousegnn.models.stgcn import create_model
from mousegnn.training.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ST-GCN inference on a skeleton npz file")
    parser.add_argument("checkpoint", type=Path, help="Path to a trained checkpoint (.pt)")
    parser.add_argument("sample", type=Path, help="Path to an npz file containing keypoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_sequence(path: Path) -> np.ndarray:
    npz = np.load(path, allow_pickle=True)
    keypoints = npz["keypoints"].astype(np.float32)  # (T, V, C)
    return keypoints


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    cfg = ckpt.get("config")
    device = args.device

    model = create_model(num_points=cfg["data"]["num_points"], num_classes=len(BEHAVIORS), in_channels=cfg["data"].get("in_channels", 2))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    transforms = Compose([NormalizeByBody()])
    sequence = transforms(load_sequence(args.sample))
    sequence = np.transpose(sequence, (2, 0, 1))  # (C, T, V)
    tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.softmax(logits, dim=1)[0]
        pred = prob.argmax().item()
    print("Prediction:", BEHAVIORS[pred])
    for idx, behavior in enumerate(BEHAVIORS):
        print(f"{behavior:>6}: {prob[idx].item():.3f}")


if __name__ == "__main__":
    main()
