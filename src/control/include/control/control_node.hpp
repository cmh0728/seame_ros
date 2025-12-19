#pragma once

#include <vector>
#include "global/global.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker.hpp"

namespace control
{

class ControlNode : public rclcpp::Node
{
public:
  // 2D 포인트 (차량 기준 좌표계)
  // x: lateral  (왼쪽 +, 오른쪽 -)
  // y: forward  (앞 +)
  struct Point2D
  {
    double x;  // lateral (left +, right -)  [m]
    double y;  // longitudinal (forward +)   [m]
  };

  ControlNode();

private:
  // ---- 콜백 ----
  void on_path(const nav_msgs::msg::Path::SharedPtr msg);

  bool compute_lookahead_target(
    const std::vector<Point2D> & path_points,
    double lookahead_distance,
    Point2D & target,
    double & actual_lookahead) const;

  double estimate_lane_slope(
    const std::vector<Point2D> & path_points) const;

  double update_speed_command(double slope, double dt);

  geometry_msgs::msg::Twist build_cmd(double curvature, double speed) const;

// --- Pure Pursuit ---
  double lookahead_distance_ = 0.45;
  double min_lookahead_      = 0.40;
  double max_lookahead_      = 0.70;
  double car_L               = 0.26;

  // --- Speed / Steer limits ---
  double base_speed_   = 30.0;
  double min_speed_    = 10.0;
  double max_speed_    = 50.0;
  double max_angular_z_ = 1.0;

  // --- Speed PID ---
  double speed_kp_       = 6.0;
  double speed_ki_       = 0.0005;
  double speed_kd_       = 0.2;
  double integral_limit_ = 5.0;

  // --- Steering filter ---
  double max_steer_rate_ = 8.0;
  double max_steer_jump_ = 1.5;
  double g_steergain     = 2.293;  // 필요하면 YAML로 뺄 수 있음

  // --- Speed rate limiting ---
  double max_speed_rate_   = 50.0;  // [speed 단위 / s], 예: 한 초에 최대 50만 변화
  double max_speed_jump_   = 20.0;  // 말도 안 되는 점프 컷할 때 쓰고 싶으면

  double prev_speed_cmd_   = 0.0;
  bool   has_prev_speed_   = false;

  // debug / topic
  bool steer_debug_      = false;
  std::string path_topic_ = "/planning/path";

  double prev_steer_cmd_ = 0.0;
  bool   has_prev_steer_ = false;

  double slope_integral_ = 0.0;
  double prev_slope_     = 0.0;
  rclcpp::Time last_update_time_;

  // 조향 변화량 필터 함수
  double filter_steering(double raw_steer, double dt);

  // 속도 변화량 필터링 
  double filter_speed(double raw_speed, double dt);


  // ---- ROS 통신 ----
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;

  // RViz에서 볼 lookahead target 마커 publisher
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr target_marker_pub_;

  // 타겟 포인트 마커 퍼블리시 함수
  void publish_target_marker(const Point2D & target, const std::string & frame_id);
  
  void LoadParam();
};

}  // namespace control
