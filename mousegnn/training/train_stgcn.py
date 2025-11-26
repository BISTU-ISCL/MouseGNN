"""Train an ST-GCN model for mouse behavior recognition."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from mousegnn.data.behaviors import BEHAVIORS
from mousegnn.datasets.skeleton_sequences import ClipSampler, Compose, NormalizeByBody, SkeletonSequenceDataset
from mousegnn.models.stgcn import create_model
from mousegnn.training.utils import accuracy, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ST-GCN on mouse skeleton sequences")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def build_dataloader(cfg: dict, split: str) -> DataLoader:
    augment = Compose(
        [
            ClipSampler(cfg["data"]["clip_len"], stride=cfg["data"].get("stride", 1)),
            NormalizeByBody(),
        ]
    )
    dataset = SkeletonSequenceDataset(Path(cfg["data"][f"{split}_dir"]), transform=augment)
    return DataLoader(
        dataset,
        batch_size=cfg["training"].get("batch_size", 8),
        shuffle=split == "train",
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> float:
    model.train()
    losses = []
    for data, labels in tqdm(loader, desc="train", leave=False):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float]:
    model.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for data, labels in tqdm(loader, desc="val", leave=False):
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            accs.append(accuracy(logits, labels))
    mean_loss = sum(losses) / max(1, len(losses))
    mean_acc = sum(accs) / max(1, len(accs))
    return mean_loss, mean_acc


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = args.device
    num_points = cfg["data"]["num_points"]
    model = create_model(num_points=num_points, num_classes=len(BEHAVIORS), in_channels=cfg["data"].get("in_channels", 2))
    model.to(device)

    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"].get("lr", 1e-3), weight_decay=cfg["training"].get("weight_decay", 5e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    output_dir = Path(cfg["training"].get("output", "runs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["training"].get("epochs", 30)):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f} time={elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "config": cfg,
                },
                output_dir / "best.pt",
            )
    print(f"Best validation accuracy: {best_acc:.3f}")


if __name__ == "__main__":
    main()
