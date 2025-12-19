cmake_minimum_required(VERSION 3.8)
project(perception)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 메시지 인터페이스 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LanePnt.msg"
  "msg/Lane.msg"
)


# 실행파일, include 경로
add_executable(CameraProcessing
  src/CamNode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../global/global.cpp
)

target_include_directories(CameraProcessing
  PRIVATE
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}   
)


ament_target_dependencies(CameraProcessing
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  yaml-cpp
  
)

target_link_libraries(CameraProcessing
  Eigen3::Eigen              
  ${OpenCV_LIBRARIES}
  yaml-cpp
)

rosidl_target_interfaces(CameraProcessing
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

ament_export_dependencies(rosidl_default_runtime)

install(TARGETS CameraProcessing
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rclcpp sensor_msgs cv_bridge OpenCV Eigen3 yaml-cpp)

ament_package()
