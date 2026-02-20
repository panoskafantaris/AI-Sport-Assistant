"""
Multi-modal detection masks for court line alignment.

Paint lines (baselines, sidelines, center) → white pixel thresholding.
Net → Canny edge detection (net is mesh/banner, not white paint).

Each mask is a binary uint8 image (0 or 255).
"""
from __future__ import annotations
import cv2
import numpy as np


def build_white_mask(
    frame: np.ndarray,
    thresh: int = 180,
    dilate_k: int = 9,
) -> np.ndarray:
    """
    White paint mask — detects painted court lines.

    Uses grayscale brightness threshold. Noise removal uses
    connected-component filtering (remove blobs < min_area)
    instead of MORPH_OPEN, which kills thin far-baseline paint
    that's only 1-2px thick after perspective compression.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    # Remove small noise blobs (< 20px area) without eroding thin lines
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    min_area = 20
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0

    # Dilate to widen attraction basin for scoring
    if dilate_k > 0:
        k_dilate = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilate_k, dilate_k))
        mask = cv2.dilate(mask, k_dilate)

    return mask


def build_edge_mask(
    frame: np.ndarray,
    low: int = 50,
    high: int = 150,
    dilate_k: int = 7,
) -> np.ndarray:
    """
    Canny edge mask — detects net and other strong edges.

    The net creates strong horizontal edges from the banner
    bottom/top and net tape. Canny captures these while the
    white mask misses them entirely.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce texture noise from net mesh
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # Canny edge detection
    edges = cv2.Canny(blur, low, high)

    # Strengthen horizontal edges (net is horizontal)
    # Use morphological closing with horizontal kernel
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_kernel)

    # Dilate for scoring tolerance
    if dilate_k > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilate_k, dilate_k))
        edges = cv2.dilate(edges, k)

    return edges


class DetectionMasks:
    """
    Container for both white and edge masks.

    Usage:
        masks = DetectionMasks.from_frame(frame)
        masks.get("white")  # for painted lines
        masks.get("edge")   # for net
    """

    def __init__(self, white: np.ndarray, edge: np.ndarray):
        self.white = white
        self.edge = edge

    @classmethod
    def from_frame(cls, frame: np.ndarray) -> DetectionMasks:
        return cls(
            white=build_white_mask(frame),
            edge=build_edge_mask(frame),
        )

    def get(self, mode: str) -> np.ndarray:
        if mode == "edge":
            return self.edge
        return self.white