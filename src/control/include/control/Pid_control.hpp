#pragma once

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float32.hpp"

namespace PidControl
{
// 차선 중심 오프셋을 받아 PID로 조향 각을 계산하고 /cmd_vel을 퍼블리시합니다.
class PidControl : public rclcpp::Node
{
public:
  PidControl();

private:
  void on_offset(const std_msgs::msg::Float32::SharedPtr msg);
  void on_heading(const std_msgs::msg::Float32::SharedPtr msg);
  void reset_if_timeout(const rclcpp::Time & now);

  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr offset_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr heading_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;

  double kp_;
  double ki_;
  double kd_;
  double max_integral_;
  double max_angular_z_;
  double linear_speed_;
  double pixel_to_meter_;
  double integral_error_;
  double prev_error_;
  double heading_error_;
  double heading_weight_;
  double last_angular_cmd_;
  rclcpp::Time last_stamp_;
  rclcpp::Duration watchdog_timeout_;
};
}  // namespace control

