import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = "auto") -> torch.device:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested with --device cuda, but no GPU is available."
            )
        return torch.device("cuda")

    if device == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_classes(data_dir: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Infers class names from dataset/train folder names.
    """
    train_dir = Path(data_dir) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    class_names = sorted(
        [path.name for path in train_dir.iterdir() if path.is_dir()]
    )
    if not class_names:
        raise RuntimeError(f"No class folders found in: {train_dir}")

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    return class_names, class_to_idx


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total


def evaluate_model(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_accuracy, all_labels, all_predictions


def print_test_metrics(labels, predictions, class_names: List[str]) -> None:
    cm = confusion_matrix(labels, predictions)
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)


def save_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
    epoch: int,
    best_val_acc: float,
    class_names: List[str],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "class_names": class_names,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_training_history(history: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
