"""
RANSAC homography matcher — global template fitting.

Strategy:
  1. Given N candidate image keypoints and 12 template world points
  2. Hypothesize: pick 4 candidates, try plausible template assignments
  3. Compute homography from 4 correspondences
  4. Verify: project ALL 12 template points → count inliers
  5. Best homography wins → derive 4 boundary corners
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np

from . import template as T
from .keypoint_detector import Candidate
from .assignment import assign_template_points


class RansacMatcher:
    """Finds the best homography mapping template → image via RANSAC."""

    def __init__(
        self,
        inlier_thresh: float = 15.0,
        n_iters: int = 500,
        min_inliers: int = 5,
    ):
        self.inlier_thresh = inlier_thresh
        self.n_iters = n_iters
        self.min_inliers = min_inliers

    def match(
        self,
        candidates: List[Candidate],
        frame_h: int,
        frame_w: int,
    ) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray]]:
        """
        Find best homography from template world → image.

        Returns:
            (corners, confidence, H_matrix)
            corners: (4,2) [TL, TR, BR, BL] in image coords, or None
            confidence: fraction of template keypoints matched
            H_matrix: 3×3 world→image homography, or None
        """
        if len(candidates) < 4:
            return None, 0.0, None

        cands = sorted(candidates, key=lambda c: c.y)
        img_pts = np.array([[c.x, c.y] for c in cands], dtype=np.float32)
        world_pts = T.KEYPOINTS_WORLD

        best_H = None
        best_inliers = 0

        for _ in range(self.n_iters):
            idx = np.random.choice(len(cands), 4, replace=False)
            sample_img = img_pts[idx]

            # Assign template points using ordering constraint
            sample_world = assign_template_points(
                sample_img, world_pts, frame_h, frame_w)
            if sample_world is None:
                continue

            # Compute homography: world → image
            H, _ = cv2.findHomography(sample_world, sample_img, 0)
            if H is None:
                continue

            # Count inliers
            inliers = self._count_inliers(H, img_pts, world_pts)
            if inliers > best_inliers:
                best_inliers = inliers
                best_H = H

        if best_H is None or best_inliers < self.min_inliers:
            return None, 0.0, None

        # Refit with all inliers for better accuracy
        best_H = self._refit_with_inliers(best_H, img_pts, world_pts)

        corners = self._project_corners(best_H)
        conf = best_inliers / T.NUM_KEYPOINTS
        return corners, conf, best_H

    def _count_inliers(
        self, H: np.ndarray,
        img_pts: np.ndarray,
        world_pts: np.ndarray,
    ) -> int:
        """Count image candidates near a projected template keypoint."""
        projected = cv2.perspectiveTransform(
            world_pts.reshape(-1, 1, 2), H).reshape(-1, 2)

        count = 0
        for proj_pt in projected:
            dists = np.linalg.norm(img_pts - proj_pt, axis=1)
            if np.min(dists) < self.inlier_thresh:
                count += 1
        return count

    def _refit_with_inliers(
        self, H: np.ndarray,
        img_pts: np.ndarray,
        world_pts: np.ndarray,
    ) -> np.ndarray:
        """Re-compute homography using all inlier correspondences."""
        projected = cv2.perspectiveTransform(
            world_pts.reshape(-1, 1, 2), H).reshape(-1, 2)

        src_list, dst_list = [], []
        for i, proj_pt in enumerate(projected):
            dists = np.linalg.norm(img_pts - proj_pt, axis=1)
            best_j = int(np.argmin(dists))
            if dists[best_j] < self.inlier_thresh:
                src_list.append(world_pts[i])
                dst_list.append(img_pts[best_j])

        if len(src_list) >= 4:
            src = np.array(src_list, dtype=np.float32)
            dst = np.array(dst_list, dtype=np.float32)
            H_new, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H_new is not None:
                return H_new
        return H

    @staticmethod
    def _project_corners(H: np.ndarray) -> np.ndarray:
        """Project the 4 boundary corners through the homography."""
        corners_world = T.BOUNDARY_CORNERS.reshape(-1, 1, 2)
        corners_img = cv2.perspectiveTransform(corners_world, H)
        return corners_img.reshape(4, 2).astype(np.float32)