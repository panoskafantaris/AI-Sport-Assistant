"""
Video loading with frame iteration.
"""
from __future__ import annotations
from typing import Generator, Iterator, Optional, Tuple
import cv2
from pathlib import Path

from ..models.frame import VideoMetadata


class VideoLoader:
    """Wraps OpenCV VideoCapture with metadata and a clean iterator."""

    def __init__(self, path: str):
        self.path = path
        self._cap: Optional[cv2.VideoCapture] = None
        self._meta: Optional[VideoMetadata] = None

    def __enter__(self) -> "VideoLoader":
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        self._meta = self._read_metadata()
        return self

    def __exit__(self, *_) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def _read_metadata(self) -> VideoMetadata:
        cap = self._cap
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoMetadata(
            width=w, height=h, fps=fps,
            total_frames=n, duration_s=n / fps,
            path=self.path,
        )

    @property
    def metadata(self) -> VideoMetadata:
        if self._meta is None:
            raise RuntimeError("VideoLoader not opened — use as context manager")
        return self._meta

    def frames(
        self,
        skip: int = 0,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[int, float, cv2.typing.MatLike], None, None]:
        """
        Yield (frame_number, timestamp_ms, frame).

        Args:
            skip:       Return every (skip+1)-th frame.
            max_frames: Stop after this many yields.
        """
        if self._cap is None:
            raise RuntimeError("VideoLoader not opened")
        fps   = self._meta.fps
        count = 0
        fn    = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if fn % (skip + 1) == 0:
                ts = (fn / fps) * 1000.0
                yield fn, ts, frame
                count += 1
                if max_frames and count >= max_frames:
                    break
            fn += 1

    def get_first_frame(self) -> Optional[cv2.typing.MatLike]:
        """Return the first frame without consuming the iterator."""
        if self._cap is None:
            raise RuntimeError("VideoLoader not opened")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self._cap.read()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame if ret else None