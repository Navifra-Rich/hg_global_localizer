cmake_minimum_required(VERSION 3.0.2)
project(placeNet)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  sensor_msgs
  roscpp
  rospy
)

catkin_package(
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(pcd_reader src/pcd_reader.cpp)
add_dependencies(pcd_reader ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pcd_reader
   ${catkin_LIBRARIES}
)