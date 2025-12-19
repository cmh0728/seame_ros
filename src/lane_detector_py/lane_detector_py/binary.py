import cv2
import numpy as np
from typing import Dict, Optional


DEFAULT_PARAMS: Dict[str, float] = {
    "clip_limit": 0.6,
    "tile_grid": 7,
    "blur_kernel": 15,
    "gray_thresh": 190,
    "sat_thresh": 200,
    "canny_low": 90,
    "canny_high": 255,
    "white_v_min": 190,
    "white_s_max": 80,
}


def _resolve_kernel(size: int) -> int:
    """Ensure the Gaussian kernel stays positive and odd-sized."""
    if size <= 1:
        return 1
    return size if size % 2 == 1 else size + 1


def create_lane_mask(bgr: np.ndarray, params: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Return a binary mask highlighting likely lane pixels.

    The optional params dict can override thresholds to tune noise suppression.
    """
    cfg = dict(DEFAULT_PARAMS)
    if params:
        cfg.update(params)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clip_limit = max(0.1, float(cfg["clip_limit"]))
    tile = int(max(2, cfg["tile_grid"]))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    v2 = clahe.apply(v)
    hsv2 = cv2.merge([h, s, v2])
    bgr2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
    kernel = _resolve_kernel(int(cfg["blur_kernel"]))
    gray_blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)

    # Tunable binary thresholds
    gray_thresh = np.clip(int(cfg["gray_thresh"]), 0, 255)
    _, binary_gray = cv2.threshold(gray_blur, gray_thresh, 255, cv2.THRESH_BINARY)

    sat_thresh = np.clip(int(cfg["sat_thresh"]), 0, 255)
    _, sat_mask = cv2.threshold(s, sat_thresh, 255, cv2.THRESH_BINARY)

    canny_low = np.clip(int(cfg["canny_low"]), 0, 255)
    canny_high = np.clip(int(cfg["canny_high"]), 0, 255)
    if canny_high <= canny_low:
        canny_high = min(255, canny_low + 1)
    edges = cv2.Canny(gray_blur, canny_low, canny_high)

    white_v_min = np.clip(int(cfg["white_v_min"]), 0, 255)
    white_s_max = np.clip(int(cfg["white_s_max"]), 0, 255)
    lower = np.array([0, 0, white_v_min], dtype=np.uint8)
    upper = np.array([180, white_s_max, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv2, lower, upper)

    combo = np.zeros_like(gray_blur)
    combo[
        (binary_gray == 255)
        | (sat_mask == 255)
        | (edges == 255)
        | (white_mask == 255)
    ] = 255
    return combo
