import os
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class GestureVideoDataset(Dataset):
    """
    PyTorch dataset for gesture videos stored as:

    dataset/
        train/
            left/
            right/
            ...
        val/
            left/
            right/
            ...
        test/
            ...

    Each item returns:
    - video tensor of shape: [num_frames, 3, height, width]
    - label tensor of shape: []
    """

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}

    def __init__(
        self,
        root_dir: str,
        class_to_idx: dict,
        num_frames: int = 16,
        frame_size: int = 224,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.root_dir}")

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"No video files found in: {self.root_dir}")

    def _build_samples(self) -> List[tuple]:
        samples = []
        for class_name in sorted(self.class_to_idx.keys()):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() in self.VIDEO_EXTENSIONS:
                    samples.append((str(file_path), self.class_to_idx[class_name]))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        video_path, label = self.samples[index]
        frames = self._load_video_frames(video_path)

        if self.transform is not None:
            frames = [self.transform(frame) for frame in frames]
        else:
            frames = [self._frame_to_tensor(frame) for frame in frames]

        video_tensor = torch.stack(frames, dim=0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return video_tensor, label_tensor

    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Loads all readable frames from a video, then uniformly samples exactly
        self.num_frames frames. If there are fewer readable frames than needed,
        the last readable frame is repeated.

        Output frame shape before tensor conversion:
        [height, width, channels] = [224, 224, 3]
        """
        capture = cv2.VideoCapture(video_path)

        all_frames = []
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            all_frames.append(frame)

        capture.release()

        if len(all_frames) == 0:
            # Robust fallback for unreadable or empty videos.
            blank_frame = np.zeros(
                (self.frame_size, self.frame_size, 3), dtype=np.uint8
            )
            all_frames = [blank_frame]

        sampled_frames = self._uniform_sample_with_padding(all_frames)
        return sampled_frames

    def _uniform_sample_with_padding(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if len(frames) >= self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            return [frames[i] for i in indices]

        padded_frames = frames.copy()
        last_frame = frames[-1]
        while len(padded_frames) < self.num_frames:
            padded_frames.append(last_frame.copy())
        return padded_frames

    @staticmethod
    def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
        """
        Converts a frame from:
        [H, W, C] -> [C, H, W]
        and normalizes values from [0, 255] to [0, 1].
        """
        frame = frame.astype(np.float32) / 255.0
        tensor = torch.from_numpy(frame).permute(2, 0, 1)
        return tensor


def get_default_transform():
    """
    Returns a simple transform function that:
    - converts frame to float tensor
    - normalizes using ImageNet mean/std (useful for ResNet-style backbones)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def transform(frame: np.ndarray) -> torch.Tensor:
        frame = frame.astype(np.float32) / 255.0
        tensor = torch.from_numpy(frame).permute(2, 0, 1)
        tensor = (tensor - mean) / std
        return tensor

    return transform
