import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import GestureVideoDataset, get_default_transform
from evaluate import run_evaluation
from model import CNNLSTM
from utils import (
    evaluate_model,
    get_device,
    infer_classes,
    save_checkpoint,
    save_training_history,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN + LSTM gesture classifier.")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset root directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames sampled per video.")
    parser.add_argument("--frame-size", type=int, default=224, help="Frame height/width.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use. Defaults to auto-selecting CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to save the best model checkpoint.",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default="checkpoints/training_history.json",
        help="Path to save training history as JSON.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    print(f"Using device: {device}")

    class_names, class_to_idx = infer_classes(args.data_dir)
    num_classes = len(class_names)

    print(f"Detected classes: {class_names}")
    print(f"Number of classes: {num_classes}")

    transform = get_default_transform()

    train_dataset = GestureVideoDataset(
        root_dir=Path(args.data_dir) / "train",
        class_to_idx=class_to_idx,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        transform=transform,
    )
    val_dataset = GestureVideoDataset(
        root_dir=Path(args.data_dir) / "val",
        class_to_idx=class_to_idx,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        transform=transform,
    )
    test_dataset = GestureVideoDataset(
        root_dir=Path(args.data_dir) / "test",
        class_to_idx=class_to_idx,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = CNNLSTM(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history_path = Path(args.history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for videos, labels in train_loader:
            # videos shape: [B, T, C, H, W] = [batch, 16, 3, 224, 224]
            # labels shape: [B]
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(outputs, dim=1)
            running_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = running_loss / total_samples if total_samples > 0 else 0.0
        train_acc = running_correct / total_samples if total_samples > 0 else 0.0

        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                checkpoint_path=str(checkpoint_path),
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_val_acc=best_val_acc,
                class_names=class_names,
            )
            print(f"Saved new best model to: {checkpoint_path}")

    save_training_history(history, str(history_path))
    print(f"Saved training history to: {history_path}")

    print("\nRunning test evaluation using the best saved model...")
    run_evaluation(
        data_dir=args.data_dir,
        checkpoint_path=str(checkpoint_path),
        split="test",
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        num_workers=args.num_workers,
        device=device,
        dataloader_override=test_loader,
    )


if __name__ == "__main__":
    main()
