#pragma once

#include "global/global.hpp"
#include <functional>
#include <string>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "perception/msg/lane.hpp"

// CameraProcessing 클래스: 이미지 구독, 차선 메시지 발행, 시각화 관리
namespace perception
{
class CameraProcessing : public rclcpp::Node
{
public:
  CameraProcessing();
  ~CameraProcessing() override;

private:
  void on_image(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg);
  void publish_lane_messages();

  std::string window_name_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_subscription_;
  rclcpp::Publisher<perception::msg::Lane>::SharedPtr lane_left_pub_;
  rclcpp::Publisher<perception::msg::Lane>::SharedPtr lane_right_pub_;
};

// 메인 파이프라인: CV::Mat → 차선 Kalman 업데이트까지 전체 흐름 제어
void Lane_detector(const cv::Mat& frame, CAMERA_DATA* camera_data);   // 메인 파이프라인

// 파라미터/맵 로더: 카메라 세팅, IPM 맵 변경 시 이 함수 구현 수정
void LoadParam(CAMERA_DATA *pst_CameraData);         // 파라미터 로드
void LoadMappingParam(CAMERA_DATA *pst_CameraData);  // IPM 맵 로드

// Lane Detection
// 히스토그램 기반 시작점 탐색 로직을 조정하려면 아래 함수군을 수정
void FindTop5MaxIndices(const int32_t* ps32_Histogram,
                        int32_t s32_MidPoint,
                        int32_t ars32_TopIndices[5],
                        bool& b_NoLane);

int32_t FindClosestToMidPoint(const int32_t points[5], int32_t s32_MidPoint);

void FindLaneStartPositions(const cv::Mat& st_Edge,
                            int32_t& s32_WindowCentorLeft,
                            int32_t& s32_WindowCentorRight,
                            bool& b_NoLaneLeft,
                            bool& b_NoLaneRight,
                            int32_t left_start_point,
                            bool has_left,
                            int32_t right_start_point,
                            bool has_right);


void SlidingWindow(const cv::Mat& st_EdgeImage,
                   const cv::Mat& st_NonZeroPosition,
                   CAMERA_LANEINFO& st_LaneInfoLeft,
                   CAMERA_LANEINFO& st_LaneInfoRight,
                   int32_t& s32_LeftWindowCentor,
                   int32_t& s32_RightWindowCentor,
                   cv::Mat& st_ResultImage);

// 모델 핏/차선 계수 계산: 다항식 차수 변경 등의 실험은 아래 부분에서
LANE_COEFFICIENT FitModel(const Point& st_Point1,
                          const Point& st_Point2,
                          bool& b_Flag);

void CalculateLaneCoefficient(CAMERA_LANEINFO& st_LaneInfo,
                              int32_t s32_Iteration,
                              int64_t s64_Threshold);

// 칼만 필터 관련 함수: 추적 안정화 파라미터를 바꾸려면 이 영역을 수정
void InitializeKalmanObject(LANE_KALMAN& st_KalmanObject);  // 칼만 필터 초기화

KALMAN_STATE CalculateKalmanState(const LANE_COEFFICIENT& st_LaneCoef,
                                  float32_t& f32_Distance,
                                  float32_t& f32_Angle);

void UpdateObservation(LANE_KALMAN& st_KalmanObject,
                       const KALMAN_STATE st_KalmanState);

void SetInitialX(LANE_KALMAN& st_KalmanObject);

void PredictState(LANE_KALMAN& st_KalmanObject);

void UpdateMeasurement(LANE_KALMAN& st_KalmanObject);

void CheckSameKalmanObject(LANE_KALMAN& st_KalmanObject,
                           KALMAN_STATE st_KalmanStateLeft);

void DeleteKalmanObject(CAMERA_DATA &pst_CameraData,
                        int32_t& s32_KalmanObjectNum,
                        int32_t s32_I);

// 시각화·보조 함수: DrawDrivingLane, MakeKalmanStateBasedLaneCoef으로 결과를 검증
void DrawDrivingLane(cv::Mat& st_ResultImage,
                     const LANE_COEFFICIENT st_LaneCoef,
                     cv::Scalar st_Color);

void MakeKalmanStateBasedLaneCoef(const LANE_KALMAN& st_KalmanObject,
                                  LANE_COEFFICIENT& st_LaneCoefficient);


// RANSAC / Kalman으로 얻은 직선 계수 → Lane 메시지로 샘플 포인트 생성
perception::msg::Lane build_lane_msg_from_coef(
    const LANE_COEFFICIENT& coef,
    int img_width,
    int img_height,
    int num_samples = 30);

// CAMERA_DATA 안의 Kalman 객체들에서 좌/우 차선 계수 뽑기
bool get_lane_coef_from_kalman(const CAMERA_DATA& cam_data,
                               LANE_COEFFICIENT& left_coef,
                               LANE_COEFFICIENT& right_coef,
                               bool& has_left,
                               bool& has_right);

bool EnforceLaneConsistencyAnchor(LANE_COEFFICIENT& left,
                                  LANE_COEFFICIENT& right,
                                  int img_height,
                                  bool left_anchor,
                                  double target_width_px,
                                  double min_width_px      = 120.0,
                                  double max_width_px      = 400.0,
                                  double max_angle_diff_deg = 8.0,
                                  double alpha_anchor_pos   = 0.1,
                                  double alpha_other_pos    = 0.7,
                                  double alpha_anchor_slope = 0.2,
                                  double alpha_other_slope  = 0.6);



}  // namespace perception