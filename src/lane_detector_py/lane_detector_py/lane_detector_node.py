#!/usr/bin/env python3
# lane detector node v1.20
from functools import partial
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
from cv_bridge import CvBridge

# Import lane detector modules
try:
    from lane_detector_py.binary import DEFAULT_PARAMS, create_lane_mask
    from lane_detector_py.birdseye import compute_homography, warp_to_top_view
    from lane_detector_py.sliding_window import fit_polynomial, sliding_window_search
    from lane_detector_py.visualization import (
        draw_lane_overlay,
        render_sliding_window_debug,
    )
except ImportError:  # pragma: no cover
    from .binary import DEFAULT_PARAMS, create_lane_mask
    from .birdseye import compute_homography, warp_to_top_view
    from .sliding_window import fit_polynomial, sliding_window_search
    from .visualization import draw_lane_overlay, render_sliding_window_debug


# Lane Detector Node Class
class LaneDetectorNode(Node):
    def __init__(self): # 노드 생성시 한번만 실행
        super().__init__('lane_detector')

        # 파라미터서버에 파라미터 등록 및 기본값 설정
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('publish_overlay_topic', '/lane/overlay')
        self.declare_parameter('publish_offset_topic', '/lane/center_offset')
        self.declare_parameter('publish_heading_topic', '/lane/heading_offset')
        self.declare_parameter('use_birdeye', True)
        self.declare_parameter('enable_visualization', True) # 디버깅용 시각화 여부 파라미터 , 기본값 False
        self.declare_parameter('lane_width_px', 650.0) # 차폭 650px 기본 설정 
        self.declare_parameter('vehicle_center_bias_px', -40.0)  # 이미지 중심 대비 차량 중심 보정값
        self.crop_size = (860, 480)
        self.last_frame_shape = None
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.lane_width_px = float(self.get_parameter('lane_width_px').get_parameter_value().double_value)
        self.lane_width_cm = 35.0  # 실제 차폭 (cm)
        self._last_logged_lane_width = None # 픽셀 차폭계산용 
        self._measured_lane_width_px = self.lane_width_px
        self.vehicle_center_bias_px = float(
            self.get_parameter('vehicle_center_bias_px').get_parameter_value().double_value
        )
        self._log_lane_width_if_needed(self.lane_width_px)

        # 버드아이용 호모그래피(예시 좌표: 해상도 640x480 전제)
        # 실제 카메라 및 트랙에 맞게 보정 필요 
        # src_points : 원본 카메라 이미지에서 변환에 사용할 4개의 점 
        # dst_points : 버드아이뷰에서 대응되는 4개의 점[x0, y0, x1, y1, x2, y2, x3, y3
        self.declare_parameter('src_points', [0.0, 400.0, 212.0, 165.0, 618.0, 165.0, 860.0, 400.0]) # 11.07 수정된 기본값 최종(카메라 꺾어서)
        self.declare_parameter('dst_points', [0.0,  480.0, 0.0,   0.0, 860.0, 0.0, 860.0, 480.0])

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.subscribe_compressed = image_topic.endswith('/compressed')
        # print(self.subscribe_compressed) # check
        # overlay_topic = self.get_parameter('publish_overlay_topic').get_parameter_value().string_value
        offset_topic = self.get_parameter('publish_offset_topic').get_parameter_value().string_value
        self.use_birdeye = self.get_parameter('use_birdeye').get_parameter_value().bool_value
        self.visualize = self.get_parameter('enable_visualization').get_parameter_value().bool_value

        self.src_pts = np.array(self.get_parameter('src_points').value, dtype=np.float32).reshape(4, 2)
        self.dst_pts = np.array(self.get_parameter('dst_points').value, dtype=np.float32).reshape(4, 2)

        # QoS: 센서데이터는 BestEffort/Depth=1이 지연/버퍼에 유리
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge() # change ros imgmsg <-> cv2

        # self.pub_overlay = self.create_publisher(Image, overlay_topic, qos) # 차선 오버레이 이미지 퍼블리셔 --> 필요없어
        self.pub_offset = self.create_publisher(Float32, offset_topic, 10)
        heading_topic = self.get_parameter('publish_heading_topic').get_parameter_value().string_value
        self.pub_heading = self.create_publisher(Float32, heading_topic, 10)

        if self.subscribe_compressed: # compressed image --> 실제 실행되는 부분
            self.sub = self.create_subscription(CompressedImage, image_topic, self.image_cb_compressed, qos)
        else: # raw image
            self.sub = self.create_subscription(Image, image_topic, self.image_cb_raw, qos)

        # 호모그래피 미리 계산(버드아이뷰 변환에 사용할 행렬)--> 프레임 계산을 줄이기 위해 한번만 실행 
        self.H, self.Hinv = self._compute_homography()

        # 디버깅용 이미지 표시 창과 마우스 콜백 설정
        self.window_name = 'lane_detector_input'
        self.control_window_src = 'homography_controls_src'
        self.control_window_dst = 'homography_controls_dst'
        self.birdeye_window = 'wrapped_img'
        self.overlay_window = 'lane_overlay'
        self.binary_control_window = 'binary_controls'

        # trackbar param
        self.homography_ui_ready = False # 트랙바 한번만 생성
        self.binary_ui_ready = False
        self._trackbar_lock = False

        self.binary_params = dict(DEFAULT_PARAMS)
        self._binary_trackbar_names = {
            'clip_limit': 'clip_limit_x10',
            'tile_grid': 'tile_grid',
            'blur_kernel': 'blur_kernel',
            'gray_thresh': 'gray_thresh',
            'sat_thresh': 'sat_thresh',
            'canny_low': 'canny_low',
            'canny_high': 'canny_high',
            'white_v_min': 'white_v_min',
            'white_s_max': 'white_s_max',
        }


        sub_type = 'CompressedImage' if self.subscribe_compressed else 'Image'
        self.get_logger().info(f'LaneDetector subscribing: {image_topic} ({sub_type})')
        # self.get_logger().info(f'Publishing overlay: {overlay_topic}, center_offset: {offset_topic}')


    #####################################  homography  ####################################################################

    def _compute_homography(self):
        return compute_homography(self.src_pts, self.dst_pts, self.use_birdeye)

    # 차폭 계산 로직 
    def _log_lane_width_if_needed(self, width: float):
        if width is None: # 양쪽 둘중 하나라도 차선못잡았을때 제외 
            return
        if self._last_logged_lane_width is None or abs(width - self._last_logged_lane_width) >= 1.0:
            self._last_logged_lane_width = width
            # self.get_logger().info(f'Estimated lane width (px): {width:.2f}')
    
    def _ensure_homography_ui(self):
        if not self.use_birdeye or self.homography_ui_ready or self.last_frame_shape is None:
            return

        ref_w, ref_h = self.last_frame_shape
        ref_w = max(1, ref_w)
        ref_h = max(1, ref_h)

        cv2.namedWindow(self.control_window_src, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.control_window_dst, cv2.WINDOW_AUTOSIZE)

        for idx in range(4):
            self._create_homography_trackbar('src', idx, 0, ref_w)
            self._create_homography_trackbar('src', idx, 1, ref_h)
            self._create_homography_trackbar('dst', idx, 0, ref_w)
            self._create_homography_trackbar('dst', idx, 1, ref_h)

        self.homography_ui_ready = True

    def _ensure_binary_ui(self):
        if not self.visualize or self.binary_ui_ready:
            return

        cv2.namedWindow(self.binary_control_window, cv2.WINDOW_AUTOSIZE)

        def _create(name: str, value: int, max_val: int, key: str):
            value = int(np.clip(value, 0, max_val))
            cv2.createTrackbar(
                name,
                self.binary_control_window,
                value,
                max_val,
                partial(self._on_binary_trackbar, key)
            )
            self._on_binary_trackbar(key, value)

        _create(self._binary_trackbar_names['clip_limit'], int(round(self.binary_params['clip_limit'] * 10)), 100, 'clip_limit')
        _create(self._binary_trackbar_names['tile_grid'], int(self.binary_params['tile_grid']), 40, 'tile_grid')
        _create(self._binary_trackbar_names['blur_kernel'], int(self.binary_params['blur_kernel']), 31, 'blur_kernel')
        _create(self._binary_trackbar_names['gray_thresh'], int(self.binary_params['gray_thresh']), 255, 'gray_thresh')
        _create(self._binary_trackbar_names['sat_thresh'], int(self.binary_params['sat_thresh']), 255, 'sat_thresh')
        _create(self._binary_trackbar_names['canny_low'], int(self.binary_params['canny_low']), 255, 'canny_low')
        _create(self._binary_trackbar_names['canny_high'], int(self.binary_params['canny_high']), 255, 'canny_high')
        _create(self._binary_trackbar_names['white_v_min'], int(self.binary_params['white_v_min']), 255, 'white_v_min')
        _create(self._binary_trackbar_names['white_s_max'], int(self.binary_params['white_s_max']), 255, 'white_s_max')

        self.binary_ui_ready = True

    def _create_homography_trackbar(self, point_type: str, idx: int, axis: int, max_val: int):
        arr = self.src_pts if point_type == 'src' else self.dst_pts
        track_name = f'{point_type}_{"x" if axis == 0 else "y"}{idx}'
        max_slider = max(1, max_val - 1)
        initial = int(np.clip(arr[idx, axis], 0, max_slider))
        arr[idx, axis] = float(initial)
        window_name = self.control_window_src if point_type == 'src' else self.control_window_dst
        cv2.createTrackbar(
            track_name,
            window_name,
            initial,
            max_slider,
            partial(self._on_homography_trackbar, point_type, idx, axis, track_name, window_name)
        )

    def _on_homography_trackbar(
        self,
        point_type: str,
        idx: int,
        axis: int,
        track_name: str,
        window_name: str,
        value: int,
    ):
        if self._trackbar_lock:
            return

        arr = self.src_pts if point_type == 'src' else self.dst_pts
        if self.last_frame_shape:
            ref_w, ref_h = self.last_frame_shape
        else:
            ref_w, ref_h = self.crop_size
        max_val = (ref_w - 1) if axis == 0 else (ref_h - 1)
        clipped = float(np.clip(value, 0, max_val))

        if clipped != value:
            try:
                self._trackbar_lock = True
                cv2.setTrackbarPos(track_name, window_name, int(clipped))
            finally:
                self._trackbar_lock = False

        arr[idx, axis] = clipped
        self.H, self.Hinv = self._compute_homography()

    def _set_binary_trackbar(self, key: str, value: int):
        name = self._binary_trackbar_names[key]
        try:
            self._trackbar_lock = True
            cv2.setTrackbarPos(name, self.binary_control_window, int(value))
        finally:
            self._trackbar_lock = False

    def _on_binary_trackbar(self, key: str, value: int):
        if self._trackbar_lock:
            return

        if key == 'clip_limit':
            value = max(1, value)
            if value != int(round(self.binary_params['clip_limit'] * 10)):
                self.binary_params['clip_limit'] = value / 10.0
            if value != cv2.getTrackbarPos(self._binary_trackbar_names[key], self.binary_control_window):
                self._set_binary_trackbar(key, value)
            return

        if key == 'tile_grid':
            value = max(2, value)
            self.binary_params['tile_grid'] = value
            if value != cv2.getTrackbarPos(self._binary_trackbar_names[key], self.binary_control_window):
                self._set_binary_trackbar(key, value)
            return

        if key == 'blur_kernel':
            value = max(1, value)
            if value % 2 == 0:
                value = value + 1 if value < 31 else value - 1
            self.binary_params['blur_kernel'] = value
            if value != cv2.getTrackbarPos(self._binary_trackbar_names[key], self.binary_control_window):
                self._set_binary_trackbar(key, value)
            return

        if key in ('gray_thresh', 'sat_thresh', 'white_v_min', 'white_s_max'):
            self.binary_params[key] = int(np.clip(value, 0, 255))
            return

        if key == 'canny_low':
            value = int(np.clip(value, 0, 254))
            self.binary_params['canny_low'] = value
            high = int(self.binary_params['canny_high'])
            if high <= value:
                high = min(255, value + 1)
                self.binary_params['canny_high'] = high
                self._set_binary_trackbar('canny_high', high)
            return

        if key == 'canny_high':
            low = int(self.binary_params['canny_low'])
            value = int(np.clip(value, low + 1, 255))
            self.binary_params['canny_high'] = value
            if value != cv2.getTrackbarPos(self._binary_trackbar_names[key], self.binary_control_window):
                self._set_binary_trackbar(key, value)
            return

    ####################################  image change cv <--> ROS ############################################################

    # cv image bridge raw
    def image_cb_raw(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return
        self._process_frame(bgr)

    # cv image bridge compressed 
    def image_cb_compressed(self, msg: CompressedImage):
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge (compressed) error: {e}')
            return
        
        # cv2.imshow('raw compressed img', bgr) # check image 
        # print(bgr.shape[0], bgr.shape[1])  # default size is 720, 1280 --> 1 is wide 0 is height
        self._process_frame(bgr)

    ################################   image processing main function ########################################################

    def _process_frame(self, bgr: np.ndarray, *, visualize: bool = None):
        viz_enabled = self.visualize if visualize is None else visualize

        # 1) 중앙 크롭: 860x480 기준으로 중앙 영역만 사용
        crop_w, crop_h = self.crop_size
        cur_h, cur_w, _ = bgr.shape
        if cur_w >= crop_w and cur_h >= crop_h:
            x0 = (cur_w - crop_w) // 2
            y0 = (cur_h - crop_h) // 2
            bgr = bgr[y0:y0 + crop_h, x0:x0 + crop_w]
        else:
            self.get_logger().warn(
                f'Incoming image smaller than crop size ({cur_w}x{cur_h} < {crop_w}x{crop_h}); skipping center crop.')

        # # 2) 상단 1/3 제거하여 하단 2/3만 사용
        # cur_h, cur_w, _ = bgr.shape
        # top_cut = cur_h // 3
        # if top_cut > 0:
        #     bgr = bgr[top_cut:, :]

        # 3) 가로가 넓을 경우 다시 중앙 정렬
        cur_h, cur_w, _ = bgr.shape
        if cur_w > crop_w:
            x0 = (cur_w - crop_w) // 2
            bgr = bgr[:, x0:x0 + crop_w]
        elif cur_w < crop_w:
            self.get_logger().warn(
                f'Incoming image narrower than crop width ({cur_w} < {crop_w}); skipping horizontal crop.')

        h, w, _ = bgr.shape
        self.last_frame_shape = (w, h)

        # track bar options
        # self._ensure_homography_ui()
        # self._ensure_binary_ui()

        # 4) 전처리 → 이진 마스크
        mask = create_lane_mask(bgr, self.binary_params)
        # print(mask.shape)  # check mask shape(2)
        if mask.ndim == 3: # if numpy array is 3 channels, convert to gray
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        

        # 5) 버드아이뷰 변환 (이진 마스크 기준)
        top = warp_to_top_view(mask, self.H) if self.H is not None else mask
        if top.ndim == 3:
            top = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        
        
        # 6) 차선 검출 부분 
        # 슬라이딩 윈도우 → 피팅
        (lx, ly), (rx, ry), window_records = sliding_window_search(top)
        left_fit_raw = fit_polynomial((lx, ly))
        right_fit_raw = fit_polynomial((rx, ry))
        left_detected = left_fit_raw is not None
        right_detected = right_fit_raw is not None

        alpha = getattr(self, 'fit_smoothing_alpha', 0.2)

        def _smooth(prev, new):
            if prev is None:
                return new.copy()
            return (1.0 - alpha) * prev + alpha * new

        left_fit = None
        if left_detected:
            raw = np.array(left_fit_raw, dtype=float)
            left_fit = _smooth(self.prev_left_fit, raw)
            self.prev_left_fit = left_fit.copy()
        elif self.prev_left_fit is not None:
            left_fit = self.prev_left_fit.copy()

        right_fit = None
        if right_detected:
            raw = np.array(right_fit_raw, dtype=float)
            right_fit = _smooth(self.prev_right_fit, raw)
            self.prev_right_fit = right_fit.copy()
        elif self.prev_right_fit is not None:
            right_fit = self.prev_right_fit.copy()

        if self.lane_width_px is not None:
            if left_fit is not None and right_fit is None:
                right_fit = left_fit.copy()
                right_fit[2] += self.lane_width_px
            elif right_fit is not None and left_fit is None:
                left_fit = right_fit.copy()
                left_fit[2] -= self.lane_width_px

        # 차 폭/센터 오프셋 계산 (픽셀 기준 → 미터 변환은 사용자 설정)
        center_offset_px = float("nan")
        y_eval = h - 1

        def _eval_fit(fit):
            if fit is None:
                return None
            return float(fit[0]*y_eval*y_eval + fit[1]*y_eval + fit[2])

        lane_center = None
        img_center = w / 2.0 + self.vehicle_center_bias_px
        have_left = left_detected and left_fit is not None
        have_right = right_detected and right_fit is not None

        def _eval_slope(fit):
            if fit is None:
                return None
            return float(2.0 * fit[0] * y_eval + fit[1])

        if have_left and have_right:
            x_left = _eval_fit(left_fit)
            x_right = _eval_fit(right_fit)
            if x_left is not None and x_right is not None:
                lane_center = (x_left + x_right) / 2.0
                self._measured_lane_width_px = float(x_right - x_left)
                self._log_lane_width_if_needed(self._measured_lane_width_px)
        elif self._measured_lane_width_px is not None:
            half_width = self._measured_lane_width_px / 2.0
            if have_left:
                x_left = _eval_fit(left_fit)
                if x_left is not None:
                    lane_center = x_left + half_width
            elif have_right:
                x_right = _eval_fit(right_fit)
                if x_right is not None:
                    lane_center = x_right - half_width

        lane_center_point_top = (lane_center, y_eval) if lane_center is not None else None

        if lane_center is not None:
            center_offset_px = float(img_center - lane_center)

        lane_slope = None
        left_slope = _eval_slope(left_fit) if have_left else None
        right_slope = _eval_slope(right_fit) if have_right else None
        if left_slope is not None and right_slope is not None:
            lane_slope = 0.5 * (left_slope + right_slope)
        else:
            lane_slope = left_slope if left_slope is not None else right_slope
        heading_offset_rad = float(np.arctan(lane_slope)) if lane_slope is not None else float("nan")

        if viz_enabled:
            # cv2.imshow(self.window_name, bgr)
            debug_view = render_sliding_window_debug(
                top, window_records, (lx, ly), (rx, ry), lane_center_point=lane_center_point_top
            )
            # cv2.imshow("mask",mask)
            cv2.imshow(self.birdeye_window, debug_view)

            fill_overlay = left_detected and right_detected
            overlay = draw_lane_overlay(
                bgr,
                top,
                self.Hinv,
                left_fit,
                right_fit,
                fill=fill_overlay,
                lane_center_point=lane_center_point_top,
                vehicle_center_px=img_center,
            )
            cv2.imshow(self.overlay_window, overlay)

            # if you want to publish overlay image, uncomment below
            # self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8'))
            cv2.waitKey(1)

        # 퍼블리시
        self.pub_offset.publish(Float32(data=center_offset_px))
        self.pub_heading.publish(Float32(data=heading_offset_rad))


    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.get_logger().info(f'Mouse click at ({x}, {y})')

def main():
    rclpy.init()
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


