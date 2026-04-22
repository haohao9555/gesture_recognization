import argparse
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".MOV"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split raw gesture videos into train/val/test, optionally crop the "
            "main person with YOLO tracking, and save sampled frames."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--zip-path",
        type=str,
        help="Path to a zip file containing class folders and videos.",
    )
    source_group.add_argument(
        "--source-dir",
        type=str,
        help="Path to a directory containing class folders and videos.",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project directory where output folders will be created.",
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
        help="Output frame size for extracted frames.",
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
    parser.add_argument(
        "--use-yolo-crop",
        action="store_true",
        help=(
            "Track people with YOLO and save cropped videos centered on the "
            "longest-visible, largest person."
        ),
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model name or path used when --use-yolo-crop is enabled.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracker config used when --use-yolo-crop is enabled.",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for person detections.",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.2,
        help="Extra padding ratio added around the tracked person box.",
    )
    parser.add_argument(
        "--min-tracked-frames",
        type=int,
        default=10,
        help=(
            "Minimum number of frames the selected person must appear in. "
            "Videos below this threshold are skipped when --use-yolo-crop is enabled."
        ),
    )
    parser.add_argument(
        "--save-debug-video",
        action="store_true",
        help=(
            "Save an annotated debug video showing all tracked people and the "
            "selected primary target."
        ),
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


def build_yolo_model(model_name: str):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError(
            "Ultralytics is required for --use-yolo-crop. "
            "Install it with `pip install ultralytics`."
        ) from error

    return YOLO(model_name)


def expand_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    width = x2 - x1
    height = y2 - y1
    pad_w = width * padding_ratio
    pad_h = height * padding_ratio

    left = max(0, int(x1 - pad_w))
    top = max(0, int(y1 - pad_h))
    right = min(frame_width, int(x2 + pad_w))
    bottom = min(frame_height, int(y2 + pad_h))
    return left, top, right, bottom


def select_primary_track(track_stats: Dict[int, Dict[str, float]]) -> Optional[int]:
    if not track_stats:
        return None

    return max(
        track_stats.keys(),
        key=lambda track_id: (
            track_stats[track_id]["count"],
            track_stats[track_id]["avg_area"],
        ),
    )


def draw_debug_overlay(
    frame: np.ndarray,
    frame_tracks: Dict[int, Tuple[int, int, int, int]],
    target_track_id: Optional[int],
) -> np.ndarray:
    annotated = frame.copy()
    for track_id, (x1, y1, x2, y2) in frame_tracks.items():
        is_target = track_id == target_track_id
        color = (0, 255, 0) if is_target else (0, 165, 255)
        thickness = 3 if is_target else 2
        label = f"id={track_id}"
        if is_target:
            label += " target"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    if target_track_id is None:
        status = "No primary target selected"
    else:
        status = f"Primary target id={target_track_id}"

    cv2.putText(
        annotated,
        status,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def crop_video_to_primary_person(
    video_path: Path,
    destination: Path,
    model,
    tracker: str,
    conf: float,
    crop_padding: float,
    min_tracked_frames: int,
    debug_video_path: Optional[Path] = None,
) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.with_suffix(".mp4")
    if debug_video_path is not None:
        debug_video_path = debug_video_path.with_suffix(".mp4")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 10.0

    track_stats: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "total_area": 0.0, "avg_area": 0.0}
    )
    per_frame_tracks: List[Dict[int, Tuple[int, int, int, int]]] = []
    frames: List[np.ndarray] = []

    while True:
        success, frame = capture.read()
        if not success:
            break

        frames.append(frame.copy())
        frame_tracks: Dict[int, Tuple[int, int, int, int]] = {}
        result = model.track(
            frame,
            persist=True,
            tracker=tracker,
            conf=conf,
            classes=[0],
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            coords = boxes.xyxy.int().cpu().tolist()
            for track_id, coord in zip(track_ids, coords):
                x1, y1, x2, y2 = coord
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                area = float(width * height)
                if area <= 0:
                    continue
                frame_tracks[track_id] = (x1, y1, x2, y2)
                track_stats[track_id]["count"] += 1
                track_stats[track_id]["total_area"] += area

        per_frame_tracks.append(frame_tracks)

    capture.release()

    for stats in track_stats.values():
        stats["avg_area"] = stats["total_area"] / stats["count"]

    target_track_id = select_primary_track(track_stats)
    if target_track_id is None or track_stats[target_track_id]["count"] < min_tracked_frames:
        return False

    first_box = None
    for frame_tracks in per_frame_tracks:
        if target_track_id in frame_tracks:
            first_box = frame_tracks[target_track_id]
            break

    if first_box is None:
        return False

    frame_height, frame_width = frames[0].shape[:2]
    left, top, right, bottom = expand_box(
        *first_box,
        frame_width=frame_width,
        frame_height=frame_height,
        padding_ratio=crop_padding,
    )
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)

    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (crop_width, crop_height),
    )
    if not writer.isOpened():
        return False

    debug_writer = None
    if debug_video_path is not None:
        debug_video_path.parent.mkdir(parents=True, exist_ok=True)
        debug_writer = cv2.VideoWriter(
            str(debug_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not debug_writer.isOpened():
            writer.release()
            return False

    last_crop = None
    for frame, frame_tracks in zip(frames, per_frame_tracks):
        if target_track_id in frame_tracks:
            left, top, right, bottom = expand_box(
                *frame_tracks[target_track_id],
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                padding_ratio=crop_padding,
            )
            crop = frame[top:bottom, left:right]
            if crop.size > 0:
                crop = cv2.resize(crop, (crop_width, crop_height))
                last_crop = crop
        if last_crop is not None:
            writer.write(last_crop)
        if debug_writer is not None:
            debug_frame = draw_debug_overlay(frame, frame_tracks, target_track_id)
            debug_writer.write(debug_frame)

    writer.release()
    if debug_writer is not None:
        debug_writer.release()
    return destination.exists() and destination.stat().st_size > 0


def prepare_video_for_split(
    video_path: Path,
    destination: Path,
    use_yolo_crop: bool,
    yolo_model,
    tracker: str,
    conf: float,
    crop_padding: float,
    min_tracked_frames: int,
    debug_video_path: Optional[Path] = None,
) -> bool:
    if not use_yolo_crop:
        copy_video(video_path, destination)
        return True

    return crop_video_to_primary_person(
        video_path=video_path,
        destination=destination,
        model=yolo_model,
        tracker=tracker,
        conf=conf,
        crop_padding=crop_padding,
        min_tracked_frames=min_tracked_frames,
        debug_video_path=debug_video_path,
    )


def main():
    args = parse_args()
    random.seed(args.seed)

    project_dir = Path(args.project_dir).resolve()
    extracted_dir = project_dir / "data" / "raw_extracted"
    split_videos_dir = project_dir / "dataset"
    split_frames_dir = project_dir / "dataset_frames"
    debug_dir = project_dir / "debug_tracks"

    if args.zip_path:
        raw_dataset_root = extract_zip(Path(args.zip_path).resolve(), extracted_dir, args.force)
    else:
        raw_dataset_root = Path(args.source_dir).resolve()
        if not raw_dataset_root.exists():
            raise FileNotFoundError(f"Source directory not found: {raw_dataset_root}")
        print(f"Using source directory: {raw_dataset_root}")

    class_to_videos = list_class_videos(raw_dataset_root)

    for target_dir in [split_videos_dir, split_frames_dir]:
        reset_dir(target_dir, args.force)
    if args.save_debug_video:
        reset_dir(debug_dir, args.force)

    yolo_model = None
    if args.use_yolo_crop:
        yolo_model = build_yolo_model(args.yolo_model)
        print(
            f"Using YOLO crop with model={args.yolo_model}, tracker={args.tracker}, "
            f"min_tracked_frames={args.min_tracked_frames}"
        )

    print("\nSplitting dataset and preparing videos...")
    kept_videos = 0
    skipped_videos = 0

    class_progress = tqdm(class_to_videos.items(), desc="Classes", unit="class")
    for class_name, videos in class_progress:
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
            video_progress = tqdm(
                split_videos,
                desc=f"{class_name}/{split_name}",
                unit="video",
                leave=False,
            )
            for video_path in video_progress:
                video_destination = split_videos_dir / split_name / class_name / video_path.name
                ok = prepare_video_for_split(
                    video_path=video_path,
                    destination=video_destination,
                    use_yolo_crop=args.use_yolo_crop,
                    yolo_model=yolo_model,
                    tracker=args.tracker,
                    conf=args.person_conf,
                    crop_padding=args.crop_padding,
                    min_tracked_frames=args.min_tracked_frames,
                    debug_video_path=(
                        debug_dir / split_name / class_name / video_path.name
                        if args.save_debug_video
                        else None
                    ),
                )

                if not ok:
                    skipped_videos += 1
                    video_progress.set_postfix_str("skipped")
                    continue

                kept_videos += 1
                sample_name = video_path.stem
                frame_output_dir = split_frames_dir / split_name / class_name / sample_name
                frames = extract_frames_from_video(
                    video_path=video_destination,
                    num_frames=args.num_frames,
                    frame_size=args.frame_size,
                )
                save_frames(frames, frame_output_dir)

    print("\nDone.")
    print(f"Prepared videos saved to: {split_videos_dir}")
    print(f"Extracted frames saved to: {split_frames_dir}")
    if args.save_debug_video:
        print(f"Debug tracking videos saved to: {debug_dir}")
    print(f"Kept videos: {kept_videos}")
    print(f"Skipped videos: {skipped_videos}")
    print("\nFolder format for extracted frames:")
    print("dataset_frames/")
    print("    train/")
    print("        class_name/")
    print("            video_name/")
    print("                frame_000.jpg")
    print("                ...")


if __name__ == "__main__":
    main()
