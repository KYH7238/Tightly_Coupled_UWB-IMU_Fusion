/**
 * Author: Yonghee Kim
 * Date: 2025-09-11
 * brief: 6D pose estimation using loosely coupled UWB/IMU fusion using filtering method(EKF/ESKF/UKF/LIEKF)
 */
#include <ros/ros.h>
#include <iostream>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <vector>
#include <queue>
#include <nlink_parser/LinktrackTagframe0.h>
#include "looselyCoupledFusionUKF.h"
#include "types.h"

using namespace std;

double dt = 0;
double beforeT = 0;
LooselyCoupled looselyFuser;
StateforEKF<double> STATE;
queue<UwbData<double>> uwbDataQueue;
queue<ImuData<double>> imuDataQueue;
bool uwbInit = false;
bool imuInit = false;
bool initialize = false;
double uwbInitTime = 0;
double imuInitTime = 0;
ros::Publisher resultPuber;

void clearQueue(queue<ImuData<double>> &q) { while (!q.empty()) q.pop(); }
void clearQueue(queue<UwbData<double>> &q) { while (!q.empty()) q.pop(); }

void publisher()
{
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "ukf";
    pose.header.stamp = ros::Time::now();

    STATE = looselyFuser.getState();

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
        if (uwbDataQueue.empty()) return;

        ImuData<double> imuData1 = imuDataQueue.front();
        ImuData<double> imuData2 = imuDataQueue.back();
        UwbData<double> uwbData = uwbDataQueue.front();

        if (imuData1.timestamp <= uwbData.timestamp && uwbData.timestamp <= imuData2.timestamp) {
            double t1 = imuData1.timestamp;
            double t2 = imuData2.timestamp;
            double t  = uwbData.timestamp;
            double delta_t = t2 - t1;
            double alpha = (delta_t == 0.0) ? 0.0 : (t - t1) / delta_t;

            imuData1.acc = (1 - alpha) * imuData1.acc + alpha * imuData2.acc;
            imuData1.gyr = (1 - alpha) * imuData1.gyr + alpha * imuData2.gyr;

            dt = t - beforeT;
            beforeT = t;

            looselyFuser.setDt(dt);
            looselyFuser.prediction(imuData1);
            looselyFuser.update(uwbData);  
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
    if (!uwbInit) {
        uwbInitTime = msg.system_time / 1000.0;
        uwbInit = true;
    }
    uwbData.timestamp = msg.system_time / 1000.0 - uwbInitTime;

    uwbData.pos[0] = msg.pos_3d[0];
    uwbData.pos[1] = msg.pos_3d[1];
    uwbData.pos[2] = -msg.pos_3d[2];
    uwbData.has_pos = true;
    uwbData.has_distance = false;

    uwbDataQueue.push(uwbData);
    processData();
}

static bool getParamFallback(const std::string& name, double& out)
{
    if (ros::param::get("~" + name, out)) return true;
    if (ros::param::get("/loosely_coupled_ukf/" + name, out)) return true;
    if (ros::param::get("/tightly_coupled_ukf/" + name, out)) return true;
    return false;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "loosely_coupled_ukf_fusion");
    ros::NodeHandle nh, ph("~");
    
    double stdV=0.0, stdW=0.0;
    if (!initialize){
        if (!getParamFallback("stdV", stdV)) {
            ROS_ERROR("Failed to get parameter stdV");
            return -1;
        }
        if (!getParamFallback("stdW", stdW)) {
            ROS_ERROR("Failed to get parameter stdW");
            return -1;
        }
        initialize = true;
        looselyFuser.setImuVar(stdV, stdW);
    }

    resultPuber = nh.advertise<geometry_msgs::PoseStamped>("/ukf", 1);
    ros::Subscriber uwbSub = nh.subscribe("/nlink_linktrack_tagframe0", 50, uwbCallback);
    ros::Subscriber imuSub = nh.subscribe("/imu/data", 200, imuCallback);

    STATE = looselyFuser.getState();
    looselyFuser.setState(STATE);

    ros::spin();  
    return 0;
}
