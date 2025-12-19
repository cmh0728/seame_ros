cmake_minimum_required(VERSION 3.8)
project(planning)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(perception REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(planning_node src/planning_node.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../global/global.cpp)

target_include_directories(planning_node
  PRIVATE
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}   
)

target_link_libraries(planning_node
  Eigen3::Eigen              
  ${OpenCV_LIBRARIES}
  yaml-cpp
)
ament_target_dependencies(planning_node
rclcpp
geometry_msgs
nav_msgs
visualization_msgs
std_msgs
yaml-cpp
perception)

install(TARGETS planning_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
