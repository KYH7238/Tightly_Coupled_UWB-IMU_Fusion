cmake_minimum_required(VERSION 2.8.3)
project(tightly_coupled)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mavros_msgs
  roscpp
  rospy
  std_msgs
  sensor_msgs
  nlink_parser
)

find_package(Eigen3 REQUIRED)

generate_messages(
   DEPENDENCIES
   std_msgs 
   geometry_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_package(
  LIBRARIES tightly_coupled_ekf tightly_coupled_eskf tightly_coupled_liekf
  CATKIN_DEPENDS geometry_msgs sensor_msgs roscpp rospy std_msgs message_runtime nlink_parser
  INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS}
)

add_library(tightly_coupled_ekf src/tightlyCoupledFusionEKF.cpp)
add_executable(ekfNode src/ekfNode.cpp)
target_link_libraries(ekfNode ${catkin_LIBRARIES} tightly_coupled_ekf)

add_library(tightly_coupled_eskf src/tightlyCoupledFusionESKF.cpp)
add_executable(eskfNode src/eskfNode.cpp)
target_link_libraries(eskfNode ${catkin_LIBRARIES} tightly_coupled_eskf)

add_library(tightly_coupled_liekf src/tightlyCoupledFusionLIEKF.cpp)
add_executable(liekfNode src/liekfNode.cpp)
target_link_libraries(liekfNode ${catkin_LIBRARIES} tightly_coupled_liekf)

add_library(tightly_coupled_ukf src/tightlyCoupledFusionUKF.cpp)
add_executable(ukfNode src/ukfNode.cpp)
target_link_libraries(ukfNode ${catkin_LIBRARIES} tightly_coupled_ukf)