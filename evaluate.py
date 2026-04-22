import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import GestureVideoDataset, get_default_transform
from model import CNNLSTM
from utils import (
    evaluate_model,
    get_device,
    infer_classes,
    load_checkpoint,
    print_test_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN + LSTM model.")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset root directory.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to the saved model checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
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
    return parser.parse_args()


def run_evaluation(
    data_dir: str,
    checkpoint_path: str,
    split: str = "test",
    batch_size: int = 8,
    num_frames: int = 16,
    frame_size: int = 224,
    num_workers: int = 0,
    device_preference: str = "auto",
    device=None,
    dataloader_override=None,
):
    if device is None:
        device = get_device(device_preference)

    print(f"Using device: {device}")

    class_names, class_to_idx = infer_classes(data_dir)

    transform = get_default_transform()
    dataset = GestureVideoDataset(
        root_dir=Path(data_dir) / split,
        class_to_idx=class_to_idx,
        num_frames=num_frames,
        frame_size=frame_size,
        transform=transform,
    )

    dataloader = dataloader_override
    if dataloader is None:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    model = CNNLSTM(num_classes=len(class_names)).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)

    if "class_names" in checkpoint:
        class_names = checkpoint["class_names"]

    criterion = nn.CrossEntropyLoss()
    loss, accuracy, labels, predictions = evaluate_model(
        model, dataloader, criterion, device
    )

    print(f"{split.capitalize()} Loss: {loss:.4f}")
    print(f"{split.capitalize()} Accuracy: {accuracy:.4f}")

    print_test_metrics(labels, predictions, class_names)


def main():
    args = parse_args()
    run_evaluation(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        split=args.split,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        num_workers=args.num_workers,
        device_preference=args.device,
    )


if __name__ == "__main__":
    main()
