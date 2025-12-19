import cv2
import numpy as np
from typing import Tuple


def compute_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    use_birdeye: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute forward/inverse homography matrices for bird's-eye projection."""
    if not use_birdeye:
        eye = np.eye(3, dtype=np.float32)
        return eye, eye

    src = np.ascontiguousarray(src_pts, dtype=np.float32).reshape(4, 2)
    dst = np.ascontiguousarray(dst_pts, dtype=np.float32).reshape(4, 2)
    H = cv2.getPerspectiveTransform(src, dst)
    Hinv = cv2.getPerspectiveTransform(dst, src)
    return H, Hinv


def warp_to_top_view(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply the bird's-eye transform (no-op if H is identity)."""
    if H is None:
        return image
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, H, (w, h))

