/**
 * Author: Yonghee Kim
 * Date: 2024-11-5
 * brief: 6D pose estimation using tightly coupled UWB/IMU fusion using UKF
 */
#include <ros/ros.h>
#include <iostream>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <vector>
#include <queue>
#include <nlink_parser/LinktrackTagframe0.h>
#include "tightlyCoupledFusionUKF.h"

using namespace std;

double dt = 0;
double beforeT = 0;
TightlyCoupled tightlyFuser;
StateforEKF<double> STATE;
queue<UwbData<double>> uwbDataQueue;
queue<ImuData<double>> imuDataQueue;
bool uwbInit = false;
bool imuInit = false;
bool initialize = false;
double uwbInitTime = 0;
double imuInitTime = 0;
Eigen::Vector3d linearAcc;
Eigen::Vector3d angularVel;
ros::Publisher resultPuber;

void clearQueue(queue<ImuData<double>> &q)
{
    while (!q.empty()) {
        q.pop();
    }
}

void clearQueue(queue<UwbData<double>> &q)
{
    while (!q.empty()) {
        q.pop();
    }
}

void publisher()
{
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "ukf";
    pose.header.stamp = ros::Time::now();
    STATE = tightlyFuser.getState();

    pose.pose.position.x = STATE.p(0);
    pose.pose.position.y = STATE.p(1);
    pose.pose.position.z = STATE.p(2);
    Eigen::Quaterniond quaternion(STATE.R);
    pose.pose.orientation.x = quaternion.x();
    pose.pose.orientation.y = quaternion.y();
    pose.pose.orientation.z = quaternion.z();
    pose.pose.orientation.w = quaternion.w();

    resultPuber.publish(pose);
}

void processData()
{   
    if (imuDataQueue.size() > 1 && !uwbDataQueue.empty()) {
        while (!uwbDataQueue.empty() &&
               (uwbDataQueue.front().timestamp < imuDataQueue.front().timestamp || 
                uwbDataQueue.front().timestamp > imuDataQueue.back().timestamp)) {
            uwbDataQueue.pop();  
        }
        ImuData<double> imuData1 = imuDataQueue.front();
        ImuData<double> imuData2 = imuDataQueue.back();
        UwbData<double> uwbData = uwbDataQueue.front();
        if (imuData1.timestamp <= uwbData.timestamp && uwbData.timestamp <= imuData2.timestamp) {
            double t1 = imuData1.timestamp;
            double t2 = imuData2.timestamp;
            double t = uwbData.timestamp;
            double alpha;
            double delta_t = t2 - t1;
            if (delta_t == 0) {
                ROS_WARN("IMU timestamps t1 and t2 are equal. Setting alpha to 0.");
                alpha = 0;
            } else {
                alpha = (t - t1) / delta_t;
            }

            imuData1.acc = (1 - alpha) * imuData1.acc + alpha * imuData2.acc;
            imuData1.gyr = (1 - alpha) * imuData1.gyr + alpha * imuData2.gyr;

            dt = t - beforeT;
            beforeT = t;

            tightlyFuser.setDt(dt);
            tightlyFuser.prediction(imuData1);
            tightlyFuser.update(uwbData);

            uwbDataQueue.pop();
            imuDataQueue.pop();
            

            publisher();
        }
    }
}

void imuCallback(const sensor_msgs::ImuConstPtr &msg)
{   
    ImuData<double> imuData;
    if (!imuInit){
        imuInitTime = msg->header.stamp.toSec();
        imuInit = true;
    }
    imuData.timestamp = msg->header.stamp.toSec() - imuInitTime;  
    imuData.gyr[0] = msg->angular_velocity.x;
    imuData.gyr[1] = msg->angular_velocity.y;
    imuData.gyr[2] = msg->angular_velocity.z;
    imuData.acc[0] = msg->linear_acceleration.x;
    imuData.acc[1] = msg->linear_acceleration.y;
    imuData.acc[2] = msg->linear_acceleration.z;

    imuDataQueue.push(imuData);
    processData();
}

void uwbCallback(const nlink_parser::LinktrackTagframe0 &msg)
{   
    UwbData<double> uwbData;
    if(!uwbInit){
        uwbInitTime = msg.system_time / 1000.00;
        uwbInit = true;
    }
    uwbData.timestamp = msg.system_time / 1000.00 - uwbInitTime;  
    for (int i = 0; i < 8; i++) {
        uwbData.distance[i] = msg.dis_arr[i];
    }
    uwbDataQueue.push(uwbData);
    processData();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "tightly_coupled_ukf_fusion");
    ros::NodeHandle nh, ph("~");
    
    double stdV, stdW;
    std::vector<double> anchorpositions_vec;
    if (!initialize){
        
        if (!ph.getParam("/tightly_coupled_ukf/stdV", stdV)) {
            ROS_ERROR("Failed to get parameter stdV");
            return -1;
        }

        if (!ph.getParam("/tightly_coupled_ukf/stdW", stdW)) {
            ROS_ERROR("Failed to get parameter stdW");
            return -1;
        }

        if (!ph.getParam("/tightly_coupled_ukf/anchorPositions", anchorpositions_vec)) {
            ROS_ERROR("Failed to get parameter anchorPositions");
            return -1;
        }

        if (anchorpositions_vec.size() == 24) {  
            Eigen::Matrix<double, 3, 8> anchorpositions;
            for (int i = 0; i < 8; i++) {
                anchorpositions(0, i) = anchorpositions_vec[i];
                anchorpositions(1, i) = anchorpositions_vec[i + 8];
                anchorpositions(2, i) = anchorpositions_vec[i + 16];
            }
            tightlyFuser.setAnchorPositions(anchorpositions);
        } else {
            ROS_ERROR("anchorPositions parameter size is incorrect");
            return -1;
        }
        initialize = true;
        tightlyFuser.setImuVar(stdV, stdW);
    }

    resultPuber = nh.advertise<geometry_msgs::PoseStamped>("/ukf", 1);
    ros::Subscriber uwbSub = nh.subscribe("/nlink_linktrack_tagframe0", 10, uwbCallback);
    ros::Subscriber imuSub = nh.subscribe("/imu/data", 10, imuCallback);
    STATE = tightlyFuser.getState();
    tightlyFuser.setState(STATE);
    ros::spin();  

    return 0;
}
