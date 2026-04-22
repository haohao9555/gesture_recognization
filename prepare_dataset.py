import argparse
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".MOV"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract videos, sample 16 frames, and split into train/val/test."
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        required=True,
        help="Path to the zip file containing class folders and videos.",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project directory where data folders will be created.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames sampled from each video.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=224,
        help="Output frame size.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing extracted/split/frame folders before running.",
    )
    return parser.parse_args()


def reset_dir(path: Path, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_dataset_root(extract_dir: Path) -> Path:
    subdirs = [path for path in extract_dir.iterdir() if path.is_dir()]
    if len(subdirs) == 1:
        nested_dirs = [path for path in subdirs[0].iterdir() if path.is_dir()]
        video_files = list(subdirs[0].rglob("*"))
        if nested_dirs and any(p.suffix in VIDEO_EXTENSIONS for p in video_files if p.is_file()):
            return subdirs[0]
    return extract_dir


def extract_zip(zip_path: Path, destination: Path, force: bool) -> Path:
    reset_dir(destination, force)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination)
    dataset_root = find_dataset_root(destination)
    print(f"Extracted zip to: {destination}")
    print(f"Detected raw dataset root: {dataset_root}")
    return dataset_root


def list_class_videos(dataset_root: Path) -> Dict[str, List[Path]]:
    class_to_videos: Dict[str, List[Path]] = {}
    for class_dir in sorted([path for path in dataset_root.iterdir() if path.is_dir()]):
        videos = sorted(
            [
                file_path
                for file_path in class_dir.iterdir()
                if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS
            ]
        )
        if videos:
            class_to_videos[class_dir.name] = videos
    if not class_to_videos:
        raise RuntimeError(f"No class folders with videos found in: {dataset_root}")
    return class_to_videos


def split_class_videos(
    videos: List[Path], train_ratio: float, val_ratio: float, test_ratio: float
) -> Dict[str, List[Path]]:
    total = len(videos)
    if total < 3:
        raise RuntimeError("Each class needs at least 3 videos to create train/val/test.")

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    # Ensure every split gets at least one sample when possible.
    if train_count == 0:
        train_count = 1
    if val_count == 0:
        val_count = 1
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > val_count:
            train_count -= 1
        else:
            val_count -= 1

    return {
        "train": videos[:train_count],
        "val": videos[train_count : train_count + val_count],
        "test": videos[train_count + val_count : train_count + val_count + test_count],
    }


def sample_frames(frames: List[np.ndarray], num_frames: int) -> List[np.ndarray]:
    if len(frames) == 0:
        raise RuntimeError("No readable frames found in video.")

    if len(frames) >= num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        return [frames[index] for index in indices]

    sampled = frames.copy()
    last_frame = frames[-1]
    while len(sampled) < num_frames:
        sampled.append(last_frame.copy())
    return sampled


def extract_frames_from_video(video_path: Path, num_frames: int, frame_size: int) -> List[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        success, frame = capture.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size))
        frames.append(frame)

    capture.release()

    if len(frames) == 0:
        blank_frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
        frames = [blank_frame]

    return sample_frames(frames, num_frames)


def save_frames(frames: List[np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_path = output_dir / f"frame_{index:03d}.jpg"
        cv2.imwrite(str(frame_path), frame_bgr)


def copy_video(video_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, destination)


def main():
    args = parse_args()
    random.seed(args.seed)

    project_dir = Path(args.project_dir).resolve()
    zip_path = Path(args.zip_path).resolve()

    extracted_dir = project_dir / "data" / "raw_extracted"
    split_videos_dir = project_dir / "dataset"
    split_frames_dir = project_dir / "dataset_frames"

    raw_dataset_root = extract_zip(zip_path, extracted_dir, args.force)
    class_to_videos = list_class_videos(raw_dataset_root)

    for target_dir in [split_videos_dir, split_frames_dir]:
        reset_dir(target_dir, args.force)

    print("\nSplitting dataset and extracting frames...")
    for class_name, videos in class_to_videos.items():
        shuffled_videos = videos.copy()
        random.shuffle(shuffled_videos)
        split_map = split_class_videos(
            shuffled_videos,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )

        print(f"\nClass: {class_name}")
        for split_name, split_videos in split_map.items():
            print(f"  {split_name}: {len(split_videos)} videos")
            for video_path in split_videos:
                video_destination = split_videos_dir / split_name / class_name / video_path.name
                copy_video(video_path, video_destination)

                sample_name = video_path.stem
                frame_output_dir = split_frames_dir / split_name / class_name / sample_name
                frames = extract_frames_from_video(
                    video_path=video_path,
                    num_frames=args.num_frames,
                    frame_size=args.frame_size,
                )
                save_frames(frames, frame_output_dir)

    print("\nDone.")
    print(f"Split videos saved to: {split_videos_dir}")
    print(f"Extracted frames saved to: {split_frames_dir}")
    print("\nFolder format for extracted frames:")
    print("dataset_frames/")
    print("    train/")
    print("        class_name/")
    print("            video_name/")
    print("                frame_000.jpg")
    print("                ...")


if __name__ == "__main__":
    main()
