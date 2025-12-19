#include "control/Pid_control.hpp"

#include <algorithm>
#include <cmath>

namespace PidControl
{
namespace
{
constexpr double kDefaultKp = 10;     // 픽셀 에러를 각속도로 변환하는 기본 비례 이득
constexpr double kDefaultKi = 0.01;
constexpr double kDefaultKd = 0.7;
constexpr double kDefaultLinearSpeed = 15.0;  // 차량 프로토콜 기준 +15가 기본 주행속도 --> 차후에 곡률에 따라 속도 조절 기능 추가 
constexpr double kDefaultMaxAngular = 1.0; //조향 최댓값
constexpr double kDefaultMaxIntegral = 1.0;
constexpr double kDefaultPixelToMeter = 0.35 / 542 ;  // user tunable scale
constexpr double kDefaultHeadingWeight = 0.2;  // 차선 각도 오프셋 가중치
constexpr double kDefaultWatchdogSec = 0.5;
}  // namespace

PidControl::PidControl()
: rclcpp::Node("pid_control"),
  integral_error_(0.0),
  prev_error_(0.0),
  heading_error_(0.0),
  last_angular_cmd_(0.0),
  last_stamp_(this->now()),
  watchdog_timeout_(rclcpp::Duration::from_seconds(kDefaultWatchdogSec))
{
  // PID 및 차량 주행 관련 기본 파라미터 선언 
  kp_ = declare_parameter("kp", kDefaultKp);
  ki_ = declare_parameter("ki", kDefaultKi);
  kd_ = declare_parameter("kd", kDefaultKd);
  linear_speed_ = declare_parameter("linear_speed", kDefaultLinearSpeed);
  max_angular_z_ = declare_parameter("max_angular_z", kDefaultMaxAngular);
  max_integral_ = declare_parameter("max_integral", kDefaultMaxIntegral);
  pixel_to_meter_ = declare_parameter("pixel_to_meter", kDefaultPixelToMeter);
  heading_weight_ = declare_parameter("heading_weight", kDefaultHeadingWeight);
  const double watchdog = declare_parameter("watchdog_timeout", kDefaultWatchdogSec); //일정시간 업데이트없으면 초기화 
  watchdog_timeout_ = rclcpp::Duration::from_seconds(watchdog);

  // /cmd_vel pub
  cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", rclcpp::QoS(10));
  
  //lane offset sub
  offset_sub_ = create_subscription<std_msgs::msg::Float32>(
    "/lane/center_offset", rclcpp::QoS(10),
    std::bind(&PidControl::on_offset, this, std::placeholders::_1));
  heading_sub_ = create_subscription<std_msgs::msg::Float32>(
    "/lane/heading_offset", rclcpp::QoS(10),
    std::bind(&PidControl::on_heading, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "PID controller initialized (kp=%.4f ki=%.4f kd=%.4f)", kp_, ki_, kd_);
}

// watchdog reset function
void PidControl::reset_if_timeout(const rclcpp::Time & now)
{
  // 일정 시간 이상 갱신이 없으면 적분/미분 항을 초기화해 급격한 제어를 방지
  if ((now - last_stamp_) > watchdog_timeout_) {
    integral_error_ = 0.0;
    prev_error_ = 0.0;
    heading_error_ = 0.0;
  }
}


// steer control callback
void PidControl::on_offset(const std_msgs::msg::Float32::SharedPtr msg)
{
  const rclcpp::Time now = this->now();
  reset_if_timeout(now);

  // dt 계산 (0으로 나누기 방지를 위해 최소값 보장)
  const double dt = std::max(1e-3, (now - last_stamp_).seconds());
  last_stamp_ = now;

  // 오프셋이 양수면 차량이 차선 중앙보다 오른쪽에 있음 (픽셀 → 미터 변환)--> - 조향 필요 
  const double raw_offset = static_cast<double>(msg->data);
  if (!std::isfinite(raw_offset)) {
    last_stamp_ = now;
    double angular_z = std::clamp(last_angular_cmd_ * 1.2, -max_angular_z_, max_angular_z_);
    last_angular_cmd_ = angular_z  ;

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = std::clamp(linear_speed_, -50.0, 50.0);
    cmd.angular.z = angular_z ;
    cmd_pub_->publish(cmd) ;
    // RCLCPP_WARN_THROTTLE(
    //   get_logger(), *this->get_clock(), 2000,
    //   "Lane offset unavailable; reusing last steering (%.3f)", angular_z);
    return;
  }

  const double error_px = raw_offset;
  const double error_m = error_px * pixel_to_meter_;
  const double heading_term = std::isfinite(heading_error_) ? heading_error_ : 0.0;
  const double combined_error = error_m + heading_weight_ * heading_term;

  // PID 적분/미분 항 계산 및 클램프
  integral_error_ = std::clamp(integral_error_ + combined_error * dt, -max_integral_, max_integral_);
  const double derivative = (combined_error - prev_error_) / dt;
  prev_error_ = combined_error;

  // PID 합산 후 각속도 제한
  double angular_z = kp_ * combined_error + ki_ * integral_error_ + kd_ * derivative;
  angular_z = std::clamp(angular_z*-1, -max_angular_z_, max_angular_z_);

  // 최종 Twist 메시지 구성 후 퍼블리시
  geometry_msgs::msg::Twist cmd;
  cmd.linear.x = std::clamp(linear_speed_, -50.0, 50.0);  // 차량 규격 범위 [-50, 50]
  cmd.angular.z = angular_z * -1 ;
  cmd_pub_->publish(cmd);
  last_angular_cmd_ = angular_z * -1 ;

  RCLCPP_DEBUG(get_logger(),
    "PID cmd: err_px=%.2f err_m=%.3f heading=%.3f combined=%.3f ang=%.3f integ=%.3f deriv=%.3f",
    error_px, error_m, heading_error_, combined_error, angular_z, integral_error_, derivative);
}

void PidControl::on_heading(const std_msgs::msg::Float32::SharedPtr msg)
{
  const double raw = static_cast<double>(msg->data);
  heading_error_ = std::isfinite(raw) ? raw : 0.0;
}
}  // namespace control

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PidControl::PidControl>());
  rclcpp::shutdown();
  return 0;
}


// speed 는 +- 50 까지, (linear x ) , steer는 +- 1 (angular z)(라디안값) +1 이 우회전, -1이 좌회전 