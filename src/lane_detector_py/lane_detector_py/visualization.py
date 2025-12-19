from typing import Dict, Iterable, Tuple, Optional

import cv2
import numpy as np

LanePoints = Tuple[np.ndarray, np.ndarray]
WindowRecords = Dict[str, Iterable[Tuple[int, int, int, int]]]


def render_sliding_window_debug(
    binary_topdown: np.ndarray,
    windows: WindowRecords,
    left_points: LanePoints,
    right_points: LanePoints,
    lane_center_point: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Return a visualization of the sliding-window search."""
    if binary_topdown.ndim == 2:
        vis = cv2.cvtColor(binary_topdown, cv2.COLOR_GRAY2BGR)
    else:
        vis = binary_topdown.copy()

    for x0, x1, y0, y1 in windows.get("left", []):
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 1)
    for x0, x1, y0, y1 in windows.get("right", []):
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)

    lx, ly = left_points
    if lx.size:
        lx_clip = np.clip(lx.astype(int), 0, vis.shape[1] - 1)
        ly_clip = np.clip(ly.astype(int), 0, vis.shape[0] - 1)
        vis[ly_clip, lx_clip] = (0, 255, 255)

    rx, ry = right_points
    if rx.size:
        rx_clip = np.clip(rx.astype(int), 0, vis.shape[1] - 1)
        ry_clip = np.clip(ry.astype(int), 0, vis.shape[0] - 1)
        vis[ry_clip, rx_clip] = (0, 255, 255)

    if lane_center_point is not None:
        cx = int(np.clip(round(lane_center_point[0]), 0, vis.shape[1] - 1))
        cy = int(np.clip(round(lane_center_point[1]), 0, vis.shape[0] - 1))
        cv2.drawMarker(vis, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

    return vis


def draw_lane_overlay(
    orig_bgr: np.ndarray,
    binary_topdown: np.ndarray,
    Hinv: np.ndarray,
    left_fit,
    right_fit,
    *,
    fill: bool = True,
    show_vehicle_center: bool = True,
    lane_center_point: Optional[Tuple[float, float]] = None,
    vehicle_center_px: Optional[float] = None,
) -> np.ndarray:
    """Project fitted lane curves onto the original image."""
    h, w = binary_topdown.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    ploty = np.arange(h)
    pts_left = None
    pts_right = None

    if left_fit is not None:
        leftx = (left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]).astype(np.int32)
        pts_left = np.stack([leftx, ploty], axis=1)
    if right_fit is not None:
        rightx = (right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]).astype(np.int32)
        pts_right = np.stack([rightx, ploty], axis=1)

    if pts_left is not None and pts_right is not None and fill:
        pts = np.vstack([pts_left, pts_right[::-1]])
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
    else:
        if pts_left is not None:
            cv2.polylines(overlay, [pts_left], False, (255, 0, 0), 5)
        if pts_right is not None:
            cv2.polylines(overlay, [pts_right], False, (0, 0, 255), 5)

    if lane_center_point is not None:
        cx = int(np.clip(round(lane_center_point[0]), 0, w - 1))
        cy = int(np.clip(round(lane_center_point[1]), 0, h - 1))
        cv2.drawMarker(overlay, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=18, thickness=2)
        cv2.circle(overlay, (cx, cy), 4, (0, 255, 255), -1)

    overlay_warped = cv2.warpPerspective(overlay, Hinv, (orig_bgr.shape[1], orig_bgr.shape[0]))
    out = cv2.addWeighted(orig_bgr, 1.0, overlay_warped, 0.4, 0)

    if show_vehicle_center:
        img_h, img_w = orig_bgr.shape[:2]
        cx = vehicle_center_px if vehicle_center_px is not None else img_w / 2.0
        center_pt = (int(round(np.clip(cx, 0, img_w - 1))), img_h - 1)
        cv2.drawMarker(out, center_pt, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.circle(out, center_pt, 5, (0, 0, 255), -1)

    return out
