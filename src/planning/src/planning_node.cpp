#include "planning/planning_node.hpp"

// Planning namespace 
namespace planning {

namespace{
// LanePoint(차량 기준 좌표계) → geometry_msgs::Point 
// LanePoint:
//   x: lateral  (왼쪽 + / 오른쪽 -)
//   y: forward  (앞쪽 +)
// ROS Point:
//   x: forward, y: left/right 

// 좌표 변환 헬퍼
geometry_msgs::msg::Point to_point(const PlanningNode::LanePoint & lane_pt, double z)
{
  geometry_msgs::msg::Point pt;
  pt.x = lane_pt.y;  // forward  (차량 전방)
  pt.y = lane_pt.x;  // lateral  (차량 좌/우)
  pt.z = z;          // 시각화용 z (보통 0 또는 약간 띄우기)
  return pt;
}
}  // namespace


PlanningNode::PlanningNode() : rclcpp::Node("planning_node")
{
  LoadParam(); // 나중에 yaml 파일로 정리, 타입 정리 
  frame_id_ = "base_link";
  // --------------------- planning parameter ---------------------------------
  path_debug_ = declare_parameter("path_debug",path_debug_); // path info 출력
  vis_marker_ = declare_parameter("vis_marker",vis_marker_); // rviz 마커 연산 + 시각화 
  // 타임스탬프 초기화 
  last_left_stamp_  = this->now();
  last_right_stamp_ = this->now();

  // ros2 qos 설정 
  auto lane_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(); // best effort
  auto vis_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable(); // reliable

  // subscriber 선언
  lane_left_sub_ = create_subscription<perception::msg::Lane>(
    "/lane/left", lane_qos,
    std::bind(&PlanningNode::on_left_lane, this, std::placeholders::_1));

  lane_right_sub_ = create_subscription<perception::msg::Lane>(
    "/lane/right", lane_qos,
    std::bind(&PlanningNode::on_right_lane, this, std::placeholders::_1));

  // path, marker publisher 선언
  path_pub_ = create_publisher<nav_msgs::msg::Path>("/planning/path", vis_qos);
  marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/planning/markers", vis_qos);

  RCLCPP_INFO(get_logger(), "Planning node ready (frame: %s)", frame_id_.c_str());
}

//################################################## on_left_lane func ##################################################//
// 좌측 차선 콜백
void PlanningNode::on_left_lane(const perception::msg::Lane::ConstSharedPtr msg)
{
  latest_left_ = msg; // 메세지 저장
  last_left_stamp_ = this->now(); // 수신 시각 갱신
  process_lanes();
}

//################################################## on_right_lane func ##################################################//
// 우측 차선 콜백
void PlanningNode::on_right_lane(const perception::msg::Lane::ConstSharedPtr msg)
{
  latest_right_ = msg; // 메세지 저장
  last_right_stamp_ = this->now(); // 수신시각 갱신 
  process_lanes();
}

//################################################## process_lanes func ##################################################//
// 좌/우 차선 최신값을 사용해 centerline + markers 생성 --> 로직 수정해야함 
void PlanningNode::process_lanes()
{
  const auto now = this->now(); // 현재 시각
  const rclcpp::Duration timeout = rclcpp::Duration::from_seconds(lane_timeout_sec_); // 0.2 초
  vis_marker_ = get_parameter("vis_marker").as_bool();
  std::vector<LanePoint> left_pts;
  std::vector<LanePoint> right_pts;

  // timeout 이내에 들어온 메시지만 유효하게 사용 (너무 오래된 차선 정보는 버림), nullptr 인지 체크 
  if (latest_left_ && (now - last_left_stamp_) <= timeout) {
    left_pts = convert_lane(latest_left_);
  }

  if (latest_right_ && (now - last_right_stamp_) <= timeout) {
    right_pts = convert_lane(latest_right_);
  }

  // 좌/우 둘 다 메세지 없으면 Path/Marker 모두 삭제 후 리턴
  if (left_pts.empty() && right_pts.empty())
  {
    // Path 비우기
    nav_msgs::msg::Path empty_path;
    empty_path.header.stamp = now;
    empty_path.header.frame_id = frame_id_;
    path_pub_->publish(empty_path);

    if(vis_marker_)
    {
    // Marker 전부 삭제 (id 0,1,2)
    visualization_msgs::msg::MarkerArray del_array;
    del_array.markers.push_back(make_delete_marker(0, "lane_left"));
    del_array.markers.push_back(make_delete_marker(1, "lane_right"));
    del_array.markers.push_back(make_delete_marker(2, "centerline"));
    marker_pub_->publish(del_array);
    }

    return;
  }

  std::vector<LanePoint> centerline; // vector 포인트 집합 
  if (!build_centerline(left_pts, right_pts, centerline)) // centerline 생성 실패
  {
    RCLCPP_DEBUG(get_logger(), "Insufficient lane data for path");

    // centerline 이 아예 안 만들어졌으면 centerline marker만 삭제
    if (vis_marker_) {
    visualization_msgs::msg::MarkerArray del_array;
    del_array.markers.push_back(make_delete_marker(2, "centerline"));
    marker_pub_->publish(del_array);
    }
    return;
  }

  // centerline 생성 성공
  // centerline 으로 path + markers 퍼블리시

  // 거의 0 값이면 , path는 중앙인데, 차가 오른쪽에 있는것, 
  // -0.03~0.08정도 나오는거면 path 자체가 오른쪽으로 밀려있는 것. --> 0.007정도 의미없는 값 나옴. path는 정상 
  path_debug_ = get_parameter("path_debug").as_bool();
  if(path_debug_){
    if (!centerline.empty()) {
    // forward ~0.5m 근처 포인트 찾기
    double target_y = 0.5;
    double best_dy = 1e9;
    double best_x = 0.0;
    for (const auto& p : centerline) {
      double dy = std::abs(p.y - target_y);
      if (dy < best_dy) {
        best_dy = dy;
        best_x = p.x;
      }
    }
    RCLCPP_INFO(get_logger(), "centerline at 0.5m: lateral=%.3f m", best_x);
    }
  }

  publish_path(centerline);
  if(vis_marker_)
  {
    publish_markers(left_pts, right_pts, centerline);
  }
}

//################################################## convert_lane func ##################################################//
// 픽셀 기반 포인트를 m 단위의 현실 프레임 단위로 변환 
std::vector<PlanningNode::LanePoint> PlanningNode::convert_lane(
  const perception::msg::Lane::ConstSharedPtr & lane_msg) const
{
  std::vector<LanePoint> rst;
  // 메시지 유효성 검사
  if (!lane_msg) {
    return rst;
  }
  
  // 메모리 미리 확보 
  rst.reserve(lane_msg->lane_points.size());

  for (const auto & pt : lane_msg->lane_points)
  {
    // IPM 픽셀 좌표 (col=c, row=r)
    const double c = static_cast<double>(pt.x);  // 0 ~ ipm_width_-1
    const double r = static_cast<double>(pt.y);  // 0 ~ ipm_height_-1

    // --- 전방 방향 X_raw (m) ---
    const double X_raw = x_max_m_ - (x_max_m_ - x_min_m_) * (r / (ipm_height_ - 1.0));
    // --- 좌우 방향 Y_raw (m) ---
    const double Y_raw = y_max_m_ - (y_max_m_ - y_min_m_) * (c / (ipm_width_ - 1.0));

    // --- 원점 오프셋 보정 (카메라 → 차량 중심으로 원점 이동 ) ---
    const double X = X_raw - origin_offset_x_m_;  // forward
    const double Y = Y_raw - origin_offset_y_m_;  // lateral

    LanePoint lane_pt;
    lane_pt.x = Y;  // lateral (왼쪽 +, 오른쪽 -)
    lane_pt.y = X;  // forward (앞 +)

    rst.push_back(lane_pt);
  }

  // 전방 방향(y=forward) 기준 오름차순 정렬
  //   → 이후 보간(sample_lane)에서 사용
  std::sort(rst.begin(), rst.end(),
    [](const LanePoint & a, const LanePoint & b) {
      return a.y < b.y;
    });

  return rst;
}

//################################################## sample_lane func ##################################################//

// 특정 전방 거리 longitudinal(y)에 대해 차선의 x(lateral)를 선형 보간하여 얻기
//   - lane: convert_lane()에서 생성된 LanePoint 리스트 (y 오름차순 정렬)
//   - longitudinal: 조회하고 싶은 전방 거리 [m]
std::optional<double> PlanningNode::sample_lane(
  const std::vector<LanePoint> & lane, double longitudinal) const
{
  if (lane.size() < 2) {
    return std::nullopt;
  }
  if (longitudinal < lane.front().y || longitudinal > lane.back().y) {
    // 요청한 y가 lane 데이터 범위 밖이면 보간 불가
    return std::nullopt;
  }

  // lower_bound로 longitudinal 이상인 첫 점을 찾음
  auto upper = std::lower_bound(
    lane.begin(), lane.end(), longitudinal,
    [](const LanePoint & pt, double value) { return pt.y < value; });

  if (upper == lane.begin()) {
    return upper->x;
  }
  if (upper == lane.end()) {
    return (lane.end() - 1)->x;
  }

  const auto & p1 = *(upper - 1);
  const auto & p2 = *upper;
  const double dy = p2.y - p1.y;
  if (std::abs(dy) < 1e-3) {
    // y가 거의 같으면 그냥 p1.x 사용
    return p1.x;
  }

  // [p1.y, p2.y] 구간에서 선형 보간
  const double ratio = (longitudinal - p1.y) / dy;
  return p1.x + ratio * (p2.x - p1.x);
}

//################################################## build_centerline func ##################################################//

// 좌/우 차선으로부터 중앙선(centerline) 생성
//
// - left, right는 각각 LanePoint 리스트 (기준 좌표계 [m])
// - y(start_offset_y_ ~ y_end) 구간을 일정 간격(resample_step_)으로 스캔하면서
//   해당 y에서의 left_x, right_x를 sample_lane()으로 얻고,
//   둘 다 있으면 평균, 한쪽만 있으면 lane_half_width_로 보정.

// 중앙선 포인트들을 생성 
bool PlanningNode::build_centerline(
  const std::vector<LanePoint> & left,
  const std::vector<LanePoint> & right,
  std::vector<LanePoint> & centerline) const
{
  centerline.clear();

  const bool has_left  = !left.empty();
  const bool has_right = !right.empty();

  // 둘 다 없으면 실패
  if (!has_left && !has_right) {
    return false;
  }

  // 플래닝에서 사용할 전체 전방 거리 범위
  const double y_start = start_offset_y_;
  const double y_end   = start_offset_y_ + max_path_length_;

  // 직전 centerline x값 + 한쪽 차선 fallback용 동적 오프셋
  double prev_center_x = 0.0;
  bool   has_prev_center = false;

  double single_lane_offset = 0.0;
  bool   offset_initialized = false;

  for (double y = y_start; y <= y_end; y += resample_step_)
  {
    std::optional<double> left_x;
    std::optional<double> right_x;

    if (has_left) {
      left_x = sample_lane(left, y);
    }
    if (has_right) {
      right_x = sample_lane(right, y);
    }

    // 둘 다 없으면 이 y에서는 centerline 못 만듦
    if (!left_x && !right_x) {
      continue;
    }

    LanePoint pt;
    pt.y = y; // forward

    if (left_x && right_x)
    {
      // ===== 1) 양쪽 차선이 모두 있는 구간: 진짜 중앙선 =====
      double center_x = (*left_x + *right_x) * 0.5;
      // pt.x = center_x;
      pt.x = center_x + centerline_offset_;

      centerline.push_back(pt);
      prev_center_x = center_x;
      has_prev_center = true;

      // 양쪽 차선이 다시 보이기 시작하면, 단일 차선 offset은 초기화
      offset_initialized = false;
    }
    else
    {
      // ===== 2) 한쪽 차선만 있는 구간 =====
      // base_lane: 한쪽 차선
      const bool use_left = (bool)left_x;
      const double base_x = use_left ? *left_x : *right_x;

      // 아직 offset이 안 정해졌다면, 경계에서 한 번 설정
      if (!offset_initialized)
      {
        if (has_prev_center)
        {
          // 직전 centerline과 base_lane 사이의 차이를 그대로 offset으로 사용
          single_lane_offset = prev_center_x - base_x;
        }
        else
        {
          // 시작부터 한쪽 차선만 있는 경우: 기존 로직처럼 lane_half_width_ 사용
          if (use_left) {
            single_lane_offset = -lane_half_width_;  // left 기준 오른쪽(-)으로
          } else {
            single_lane_offset = +lane_half_width_;  // right 기준 왼쪽(+)으로
          }
        }
        offset_initialized = true;
      }

      double center_x = base_x + single_lane_offset;
      // pt.x = center_x;
      pt.x = center_x + centerline_offset_;

      centerline.push_back(pt);
      prev_center_x = center_x;
      has_prev_center = true;
    }
  }

  return !centerline.empty();
}



//################################################## vis : publish_path func ##################################################//

// 중앙선(centerline) → nav_msgs::Path 로 퍼블리시
//
// - Path의 각 포인트는 PoseStamped
// - position.x: forward, position.y: lateral 로 매핑
// - orientation: 다음 포인트와의 방향(yaw)을 이용해 계산

// path 메시지 생성 및 퍼블리시 
void PlanningNode::publish_path(const std::vector<LanePoint> & centerline)
{
  nav_msgs::msg::Path path_msg;
  path_msg.header.stamp = now(); 
  path_msg.header.frame_id = frame_id_;

  // centerline 포인트 하나하나를 PoseStamped로 변환
  for (size_t i = 0; i < centerline.size(); ++i)
  {
    const auto & pt = centerline[i];

    geometry_msgs::msg::PoseStamped pose;
    pose.header = path_msg.header;

    //PoseStamped로 포인트 이동 
    // 좌표계 매핑: x=forward, y=lateral
    pose.pose.position.x = pt.y;  // forward
    pose.pose.position.y = pt.x;  // lateral
    pose.pose.position.z = marker_z_; // default 0.0

    // yaw 계산 (다음 점과의 방향)
    double yaw = 0.0;
    if (i + 1 < centerline.size())
    {
      const auto & next = centerline[i + 1];
      // (Δx = next.y - pt.y, Δy = next.x - pt.x)
      yaw = std::atan2(next.x - pt.x, next.y - pt.y);
    }

    pose.pose.orientation.x = 0.0;
    pose.pose.orientation.y = 0.0;
    pose.pose.orientation.z = std::sin(yaw * 0.5);
    pose.pose.orientation.w = std::cos(yaw * 0.5);

    path_msg.poses.push_back(pose);
  }

  path_pub_->publish(path_msg);
}

//################################################## vis : publish_markers func ##################################################//

// 좌/우 차선 + 중앙선 → MarkerArray (LINE_STRIP) 로 퍼블리시
//
// - left/right/centerline 각각 다른 색/ID로 표시
void PlanningNode::publish_markers(
  const std::vector<LanePoint> & left,
  const std::vector<LanePoint> & right,
  const std::vector<LanePoint> & centerline)
{
  visualization_msgs::msg::MarkerArray array;

  // 왼쪽 차선 marker
  if (!left.empty()) {
    array.markers.push_back(make_marker(left, 0, "lane_left", 0.2, 0.6, 1.0));
  } else {
    array.markers.push_back(make_delete_marker(0, "lane_left"));
  }

  // 오른쪽 차선 marker
  if (!right.empty()) {
    array.markers.push_back(make_marker(right, 1, "lane_right", 1.0, 0.6, 0.2));
  } else {
    array.markers.push_back(make_delete_marker(1, "lane_right"));
  }

  // 중앙선 marker
  if (!centerline.empty()) {
    array.markers.push_back(make_marker(centerline, 2, "centerline", 0.1, 1.0, 0.2));
  } else {
    array.markers.push_back(make_delete_marker(2, "centerline"));
  }

  marker_pub_->publish(array);
}

//################################################## vis : make_marker func ##################################################//

// 실제 차선 / 중앙선 LineStrip Marker 생성
visualization_msgs::msg::Marker PlanningNode::make_marker(
  const std::vector<LanePoint> & lane,
  int id,
  const std::string & ns,
  double r, double g, double b) const
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame_id_;
  marker.header.stamp = now();
  marker.ns = ns;
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = 0.05;  // 선 두께
  marker.color.a = 1.0;
  marker.color.r = r;
  marker.color.g = g;
  marker.color.b = b;
  marker.pose.orientation.w = 1.0;

  marker.points.reserve(lane.size());
  for (const auto & pt : lane) {
    marker.points.push_back(to_point(pt, marker_z_));
  }

  return marker;
}

//################################################## vis : make_delete_marker func ##################################################//

// 기존 Marker 삭제용 (id/namespace로 해당 marker 제거)
visualization_msgs::msg::Marker PlanningNode::make_delete_marker(
  int id,
  const std::string & ns) const
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame_id_;
  marker.header.stamp = now();
  marker.ns = ns;
  marker.id = id;
  marker.action = visualization_msgs::msg::Marker::DELETE;
  return marker;
}

//################################################## loadparam func ##################################################//

// YAML palnning 설정을 로드하고 기본 상태를 초기화
void PlanningNode::LoadParam()
{
  try
  {
    YAML::Node node = YAML::LoadFile("src/Params/config.yaml");
    std::cout << "Loading Planning Parameter from YAML File..." << std::endl;

    if (node["x_min_m"])        x_min_m_        = node["x_min_m"].as<double>();
    if (node["x_max_m"])        x_max_m_        = node["x_max_m"].as<double>();
    if (node["y_min_m"])        y_min_m_        = node["y_min_m"].as<double>();
    if (node["y_max_m"])        y_max_m_        = node["y_max_m"].as<double>();
    if (node["ipm_width"])      ipm_width_      = node["ipm_width"].as<double>();
    if (node["ipm_height"])     ipm_height_     = node["ipm_height"].as<double>();

    if (node["origin_offset_x_m"])
      origin_offset_x_m_ = node["origin_offset_x_m"].as<double>();
    if (node["origin_offset_y_m"])
      origin_offset_y_m_ = node["origin_offset_y_m"].as<double>();

    if (node["lane_half_width"])
      lane_half_width_   = node["lane_half_width"].as<double>();
    if (node["resample_step"])
      resample_step_     = node["resample_step"].as<double>();

    if (node["max_path_length"])
      max_path_length_   = node["max_path_length"].as<double>();
    if (node["start_offset_y"])
      start_offset_y_    = node["start_offset_y"].as<double>();

    if (node["marker_z"])
      marker_z_          = node["marker_z"].as<double>();
    if (node["lane_timeout_sec"])
      lane_timeout_sec_  = node["lane_timeout_sec"].as<double>();

    if (node["path_debug"])
      path_debug_         = node["path_debug"].as<bool>();

    if (node["centerline_offset"])
      centerline_offset_ = node["centerline_offset"].as<double>();


    std::cout << "Success to Load Planning Parameter!" << std::endl;
  }
  catch (const std::exception & e)
  {
    std::cerr << "[PlanningNode] Failed to load config.yaml: "
              << e.what()
              << " (use built-in defaults)" << std::endl;
  }
}



}  // namespace planning



//################################################## main func ##################################################//

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<planning::PlanningNode>());
  rclcpp::shutdown();
  return 0;
}
