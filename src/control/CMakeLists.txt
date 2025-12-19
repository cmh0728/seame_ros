cmake_minimum_required(VERSION 3.8)
project(control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)




add_executable(control_node 
  src/control_node.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../global/global.cpp
  )

add_executable(pid_control
  src/Pid_control.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../global/global.cpp
)
target_include_directories(control_node  PRIVATE include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS} )

target_include_directories(pid_control  PRIVATE include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS} )


ament_target_dependencies(control_node
  rclcpp
  geometry_msgs
  nav_msgs
  std_msgs
  OpenCV
  visualization_msgs
  yaml-cpp)

ament_target_dependencies(pid_control
  rclcpp
  geometry_msgs
  std_msgs
  visualization_msgs
  yaml-cpp
  nav_msgs
  OpenCV
)

target_link_libraries(control_node
  Eigen3::Eigen              
  ${OpenCV_LIBRARIES}
  yaml-cpp
)

target_link_libraries(pid_control
  Eigen3::Eigen              
  ${OpenCV_LIBRARIES}
  yaml-cpp
)

install(TARGETS 
  control_node 
  pid_control
  DESTINATION lib/${PROJECT_NAME})

ament_package()
