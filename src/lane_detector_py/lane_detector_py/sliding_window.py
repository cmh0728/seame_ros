from typing import Dict, List, Tuple, Optional

import numpy as np

WindowRecords = Dict[str, List[Tuple[int, int, int, int]]]
LanePoints = Tuple[np.ndarray, np.ndarray]

_last_left_base: Optional[int] = None
_last_right_base: Optional[int] = None


def _apply_history(
    candidate: Optional[int],
    last_value: Optional[int],
    max_jump: int,
) -> Tuple[Optional[int], Optional[int]]:
    """이전 프레임과 비교해 허용범위 내 베이스만 채택."""
    if candidate is None:
        return None, last_value
    if last_value is None or abs(candidate - last_value) <= max_jump:
        return candidate, candidate
    return None, last_value

# histogram 피크찾기
def _hist_peaks(
    binary_topdown: np.ndarray,
    margin_bottom: float = 0.4,  # 하단 0.4영역에서 피크를 찾음
    min_peak_value: int = 50,  # 최소 픽셀 수 조건
    max_base_jump: int = 80,  # 이전 프레임 대비 허용 오차
):
    """Find histogram peaks (base positions) near the bottom of the binary image."""
    # 바닥 영역 히스토그램을 통해 왼쪽/오른쪽 차선의 시작 x좌표를 추정
    h, w = binary_topdown.shape
    y0 = int(h * (1.0 - margin_bottom))
    region = binary_topdown[y0:, :]
    histogram = np.sum(region // 255, axis=0)

    mid = w // 2  # 이미지 중심 픽셀값

    left_hist = histogram[:mid]
    right_hist = histogram[mid:]
    leftx_base = np.argmax(left_hist) if left_hist.any() else None  # 흰색 픽셀이 하나도 없으면 None 반환
    rightx_base = np.argmax(right_hist) + mid if right_hist.any() else None

    left_peak_value = int(left_hist[leftx_base]) if leftx_base is not None else 0
    right_peak_value = int(right_hist[rightx_base - mid]) if rightx_base is not None else 0

    # 최소픽셀값 디버깅 
    # print(f"[sliding_window] left peak: {left_peak_value}, right peak: {right_peak_value}",flush=True,)

    # 최소픽셀조건보다 작으면 none 처리 
    if leftx_base is not None and left_peak_value < min_peak_value:
        leftx_base = None
    if rightx_base is not None and right_peak_value < min_peak_value:
        rightx_base = None

    global _last_left_base, _last_right_base
    leftx_base, _last_left_base = _apply_history(leftx_base, _last_left_base, max_base_jump)
    rightx_base, _last_right_base = _apply_history(
        rightx_base, _last_right_base, max_base_jump
    )
    return leftx_base, rightx_base


def sliding_window_search(
    binary_topdown: np.ndarray,
    nwindows: int = 12,
    window_width: int = 50,  # 윈도우 가로폭 픽셀값
    minpix: int = 30,  # 윈도우 내 유효 픽셀 수 최솟값
    min_peak_value: int = 50,  # 히스토그램 피크 최소 픽셀 수
    center_guard_px: int = 150,  # 중앙 기준 좌/우 윈도우 침범 방지 여유폭
    max_base_jump_px: int = 80,  # 이전 프레임 대비 허용 오차
    ) -> Tuple[LanePoints, LanePoints, WindowRecords]:
    """Collect left/right lane pixels with a histogram-guided sliding-window search."""
    # 바이너리 버드뷰 이미지 좌표를 준비
    h, w = binary_topdown.shape
    nonzero = binary_topdown.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # 히스토그램으로 첫 윈도우의 중앙 좌표를 잡음
    leftx_base, rightx_base = _hist_peaks(
        binary_topdown,
        margin_bottom=0.4,
        min_peak_value=min_peak_value,
        max_base_jump=max_base_jump_px,
    )
    # print(f"Left base: {leftx_base}, Right base: {rightx_base}") # Left base: 146, Right base: 688 , 차이는 542픽셀

    leftx_current, rightx_current = leftx_base, rightx_base

    # 슬라이딩 윈도우 높이 및 결과 저장 리스트 초기화
    window_height = h // nwindows if nwindows else h
    mid = w // 2
    guard = max(0, int(center_guard_px))
    left_x_max = max(1, mid - guard)
    right_x_min = min(w - 1, mid + guard)
    left_lane_inds = []
    right_lane_inds = []
    window_records: WindowRecords = {"left": [], "right": []}

    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height

        def _gather(x_center: int, allowed_range: Tuple[int, int]):
            # 현재 윈도우 중심을 기준으로 유효한 픽셀을 모으고 중심을 새로 계산
            if x_center is None:
                return None, None, None, None, np.array([], dtype=int), None
            allow_min, allow_max = allowed_range
            if allow_max - allow_min <= 1:
                return None, None, None, None, np.array([], dtype=int), None
            win_x_low = max(allow_min, x_center - window_width // 2)
            win_x_high = min(allow_max, x_center + window_width // 2)
            if win_x_low >= win_x_high:
                return None, None, None, None, np.array([], dtype=int), None
            good_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_x_low)
                & (nonzerox < win_x_high)
            ).nonzero()[0]
            new_center = x_center
            if len(good_inds) > minpix:
                new_center = int(np.mean(nonzerox[good_inds]))
                new_center = int(np.clip(new_center, allow_min, allow_max - 1))
            return win_x_low, win_x_high, win_y_low, win_y_high, good_inds, new_center

        guard_active = guard and window < 4

        if leftx_current is not None:
            # 왼쪽 차선 윈도우 추적
            left_range = (0, left_x_max) if guard_active else (0, w)
            lx0, lx1, ly0, ly1, good, leftx_current = _gather(leftx_current, left_range)
            if good.size:
                left_lane_inds.append(good)
            if lx0 is not None and lx1 is not None:
                window_records["left"].append((lx0, lx1, ly0, ly1))

        if rightx_current is not None:
            # 오른쪽 차선 윈도우 추적
            right_range = (right_x_min, w) if guard_active else (0, w)
            rx0, rx1, ry0, ry1, good, rightx_current = _gather(rightx_current, right_range)
            if good.size:
                right_lane_inds.append(good)
            if rx0 is not None and rx1 is not None:
                window_records["right"].append((rx0, rx1, ry0, ry1))

    left_inds = (
        np.concatenate(left_lane_inds) if len(left_lane_inds) else np.array([], dtype=int)
    )
    right_inds = (
        np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([], dtype=int)
    )

    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]
    return (leftx, lefty), (rightx, righty), window_records


def fit_polynomial(points: LanePoints):
    """Fit a second-order polynomial x(y) = ay^2 + by + c if enough points exist."""
    # 충분한 포인트가 있을 때만 2차 곡선을 피팅
    x, y = points
    if len(x) < 50:
        return None
    return np.polyfit(y, x, 2)


