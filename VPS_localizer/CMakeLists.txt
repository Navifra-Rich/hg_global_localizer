cmake_minimum_required(VERSION 3.0.2)
project(VPS_localizer)

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
  include
  ${catkin_INCLUDE_DIRS}
)
