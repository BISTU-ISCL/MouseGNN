"""Simple keyboard-driven video annotator for mouse behaviors.

The annotator streams frames from a video file and lets you toggle the active
behavior using single-key shortcuts. When you switch behaviors the tool
captures a segment ``(start_frame, end_frame, behavior)`` that is written to a
CSV file for training.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from mousegnn.data.behaviors import BEHAVIORS, behavior_to_index

KEY_BINDINGS: Dict[int, str] = {
    ord("w"): "walk",
    ord("c"): "circle",
    ord("s"): "sniff",
    ord("r"): "rear",
    ord("k"): "curl",
}


@dataclass
class Segment:
    start: int
    end: int
    behavior: str


class BehaviorAnnotator:
    """Interactive OpenCV annotator.

    Usage:
        annotator = BehaviorAnnotator(video_path, output_csv)
        annotator.run()
    """

    def __init__(self, video_path: Path, output_csv: Path, start_frame: int = 0):
        self.video_path = Path(video_path)
        self.output_csv = Path(output_csv)
        self.start_frame = start_frame
        self.segments: List[Segment] = []
        self.active_label: Optional[str] = None
        self.active_start: Optional[int] = None

    def _record_segment(self, end_frame: int) -> None:
        if self.active_label is None or self.active_start is None:
            return
        self.segments.append(Segment(self.active_start, end_frame, self.active_label))

    def _draw_overlay(self, frame, frame_idx: int) -> None:
        status = f"Frame: {frame_idx} | Active: {self.active_label or 'none'}"
        cv2.putText(frame, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y = 48
        for key, behavior in sorted(KEY_BINDINGS.items()):
            label = f"[{chr(key)}] {behavior}"
            cv2.putText(frame, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 18
        cv2.putText(frame, "[space] clear segment  [q] save & quit", (12, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def run(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        frame_idx = self.start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        while True:
            ret, frame = cap.read()
            if not ret:
                self._record_segment(frame_idx - 1)
                break

            self._draw_overlay(frame, frame_idx)
            cv2.imshow("mouse-behavior-annotator", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in KEY_BINDINGS:
                # finalize previous label segment
                self._record_segment(frame_idx - 1)
                # start new label
                self.active_label = KEY_BINDINGS[key]
                self.active_start = frame_idx
            elif key == ord(" "):
                self._record_segment(frame_idx - 1)
                self.active_label = None
                self.active_start = None
            elif key == ord("q"):
                self._record_segment(frame_idx)
                break

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()
        self.save()

    def save(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video", "start_frame", "end_frame", "behavior", "behavior_index"])
            writer.writeheader()
            for seg in self.segments:
                writer.writerow(
                    {
                        "video": str(self.video_path),
                        "start_frame": seg.start,
                        "end_frame": seg.end,
                        "behavior": seg.behavior,
                        "behavior_index": behavior_to_index()[seg.behavior],
                    }
                )
        print(f"Saved {len(self.segments)} segments to {self.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate mouse behaviors in a video with keyboard shortcuts.")
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument("output", type=Path, help="Path to write the CSV annotations.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index to start labeling from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotator = BehaviorAnnotator(args.video, args.output, start_frame=args.start_frame)
    annotator.run()


if __name__ == "__main__":
    main()
