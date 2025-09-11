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
  LIBRARIES tightly_coupled_ekf tightly_coupled_eskf tightly_coupled_ukf tightly_coupled_liekf loosely_coupled_ekf loosely_coupled_eskf loosely_coupled_ukf 2tag_ekf
  CATKIN_DEPENDS geometry_msgs sensor_msgs roscpp rospy std_msgs message_runtime nlink_parser
  INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS}
)

add_library(tightly_coupled_ekf src/tightlyCoupledFusionEKF.cpp)
add_executable(tightlyEkfNode src/tightlyEkfNode.cpp)
target_link_libraries(tightlyEkfNode ${catkin_LIBRARIES} tightly_coupled_ekf)

add_library(tightly_coupled_eskf src/tightlyCoupledFusionESKF.cpp)
add_executable(tightlyEskfNode src/tightlyEskfNode.cpp)
target_link_libraries(tightlyEskfNode ${catkin_LIBRARIES} tightly_coupled_eskf)

add_library(tightly_coupled_liekf src/tightlyCoupledFusionLIEKF.cpp)
add_executable(tightlyLiekfNode src/tightlyLiekfNode.cpp)
target_link_libraries(tightlyLiekfNode ${catkin_LIBRARIES} tightly_coupled_liekf)

add_library(tightly_coupled_ukf src/tightlyCoupledFusionUKF.cpp)
add_executable(ukfNode src/tightlyUkfNode.cpp)
target_link_libraries(ukfNode ${catkin_LIBRARIES} tightly_coupled_ukf)

add_library(loosely_coupled_ekf src/looselyCoupledFusionEKF.cpp)
add_executable(looselyEkfNode src/looselyEkfNode.cpp)
target_link_libraries(looselyEkfNode ${catkin_LIBRARIES} loosely_coupled_ekf)

add_library(loosely_coupled_eskf src/looselyCoupledFusionESKF.cpp)
add_executable(looselyEskfNode src/looselyEskfNode.cpp)
target_link_libraries(looselyEskfNode ${catkin_LIBRARIES} loosely_coupled_eskf)

add_library(loosely_coupled_ukf src/looselyCoupledFusionUKF.cpp)
add_executable(looselyUkfNode src/looselyUkfNode.cpp)
target_link_libraries(looselyUkfNode ${catkin_LIBRARIES} loosely_coupled_ukf)