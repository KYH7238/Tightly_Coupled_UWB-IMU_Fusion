/**
 * Author: Yonghee Kim
 * Date: 2025-09-11
 * brief: 6D pose estimation using loosely coupled UWB/IMU fusion using filtering method(EKF/ESKF/UKF/LIEKF)
 */
#include "looselyCoupledFusionESKF.h" 
#include <Eigen/Dense>
#include <Eigen/Geometry>

LooselyCoupled::LooselyCoupled()
{
    covP.setZero();
    covQ.setIdentity()*0.001;
    covR = Eigen::MatrixXd::Identity(3,3)*0.08;

    vecZ = Eigen::VectorXd::Zero(3);
    vecH = Eigen::VectorXd::Zero(3);
    jacobianMatF.setIdentity();
    jacobianMatH = Eigen::MatrixXd::Zero(3,15);
    _g << 0, 0, 9.81;

    TOL = 1e-9;
    dt = 0.0;
    STATE.p.setZero();
    STATE.q = Eigen::Quaterniond::Identity();
    STATE.v.setZero();
    STATE.a_b.setZero();
    STATE.w_b.setZero();
}

LooselyCoupled::~LooselyCoupled() {}

void LooselyCoupled::setDt(const double delta_t){
    dt = delta_t;
}

void LooselyCoupled::setZ(const UwbData<double> &currUwbData){
    if(currUwbData.has_pos){
        vecZ = currUwbData.pos.cast<double>();
        vecH = Eigen::Vector3d::Zero();
        covR = Eigen::Matrix3d::Identity()*0.08;
        jacobianMatH = Eigen::MatrixXd::Zero(3,15);
    }
}

void LooselyCoupled::setImuVar(const double stdV, const double stdW){
    covQ.block<3,3>(9,9)  = stdV*Eigen::Matrix3d::Identity();
    covQ.block<3,3>(12,12)= stdW*Eigen::Matrix3d::Identity();
}

Eigen::Matrix3d LooselyCoupled::vectorToSkewSymmetric(const Eigen::Vector3d &vector){
    Eigen::Matrix3d Rot;
    Rot << 0, -vector.z(), vector.y(),
           vector.z(), 0, -vector.x(),
           -vector.y(), vector.x(), 0;
    return Rot;
}

Eigen::Matrix3d LooselyCoupled::Exp(const Eigen::Vector3d &omega){
    double angle = omega.norm();
    Eigen::Matrix3d Rot;
    if (angle < TOL){
        Rot = Eigen::Matrix3d::Identity();
    } else {
        Eigen::Vector3d axis = omega/angle;
        double c = cos(angle);
        double s = sin(angle);
        Rot = c*Eigen::Matrix3d::Identity() + (1 - c)*axis*axis.transpose() + s*vectorToSkewSymmetric(axis);
    }
    return Rot;
}

Eigen::Quaterniond LooselyCoupled::getQuaFromAA(const Eigen::Vector3d &vec){
    double theta = vec.norm();
    if(theta < 1e-12) return Eigen::Quaterniond::Identity();
    Eigen::Vector3d axis = vec/theta;
    double w = cos(0.5*theta);
    double s = sin(0.5*theta);
    return Eigen::Quaterniond(w, axis.x()*s, axis.y()*s, axis.z()*s).normalized();
}

void LooselyCoupled::motionModelJacobian(const ImuData<double> &imu_data){
    Eigen::Matrix3d Rot = STATE.q.toRotationMatrix();
    jacobianMatF.setIdentity();
    jacobianMatF.block<3,3>(0,6) = dt*Eigen::Matrix3d::Identity();
    jacobianMatF.block<3,3>(6,9) = -Rot*dt;
    jacobianMatF.block<3,3>(3,12)= -dt*Eigen::Matrix3d::Identity();
    jacobianMatF.block<3,3>(6,3) = -Rot*vectorToSkewSymmetric(imu_data.acc-STATE.a_b)*dt;
}

void LooselyCoupled::motionModel(const ImuData<double> &imu_data){
    Eigen::Matrix3d Rot = STATE.q.toRotationMatrix();
    Eigen::Vector3d accWorld = Rot*(imu_data.acc - STATE.a_b) + _g;
    STATE.p += STATE.v*dt + 0.5*accWorld*dt*dt;
    STATE.v += accWorld*dt;
    Eigen::Quaterniond deltaQ = getQuaFromAA((imu_data.gyr - STATE.w_b)*dt);
    STATE.q = STATE.q * deltaQ;
    STATE.q.normalize();
}

void LooselyCoupled::prediction(const ImuData<double> &imu_data){
    motionModelJacobian(imu_data);
    motionModel(imu_data);
    covP = jacobianMatF*covP*jacobianMatF.transpose() + covQ;
}

void LooselyCoupled::measurementModel(){
    vecH = STATE.p;  
}

void LooselyCoupled::measurementModelJacobian(){
    jacobianMatH.setZero();
    jacobianMatH.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
}

StateforESKF<double> LooselyCoupled::correction(){
    measurementModel();
    measurementModelJacobian();
    Eigen::Matrix3d residualCov = jacobianMatH*covP*jacobianMatH.transpose() + covR;
    if(residualCov.determinant() == 0 || !residualCov.allFinite()){
        std::cerr << "residualCov singular/NaN" << std::endl;
        return STATE;
    }
    Eigen::Matrix<double,15,3> K = covP*jacobianMatH.transpose()*residualCov.inverse();
    Eigen::Matrix<double,15,1> updateState = K*(vecZ-vecH);

    STATE.p += updateState.segment<3>(0);
    Eigen::Quaterniond deltaQ = getQuaFromAA(updateState.segment<3>(3));
    STATE.q = STATE.q * deltaQ;
    STATE.q.normalize();
    STATE.v += updateState.segment<3>(6);
    STATE.a_b += updateState.segment<3>(9);
    STATE.w_b += updateState.segment<3>(12);

    Eigen::Matrix<double,15,15> IKH = Eigen::Matrix<double,15,15>::Identity() - K*jacobianMatH;
    covP = IKH*covP*IKH.transpose() + K*covR*K.transpose();

    return STATE;
}

void LooselyCoupled::setState(StateforESKF<double> &State){
    STATE = State;
}

StateforESKF<double> LooselyCoupled::getState(){
    return STATE;
}
