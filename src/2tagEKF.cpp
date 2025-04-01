/**
 * Author: Yonghee Kim
 * Date: 2025-04-01
 * brief: 6D pose estimation using tightly coupled UWB/IMU fusion using filtering method(EKF/ESKF/UKF/LIEKF)
 */
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "2tag.h" 

TightlyCoupled::TightlyCoupled()
{
    covP.setZero();
    covQ.setIdentity()*0.001;
    covR.setIdentity()*0.08;
    vecZ.setZero();
    vecH.setZero();

    jacobianMatF.setIdentity();
    jacobianMatH.setZero();
    anchorPositions.setZero();
    _g << 0, 0, 9.81;

    TOL =1e-9;
    dt = 0;
    STATE.p <<0, 0, 0;
    STATE.R.setIdentity();

    STATE.v.setZero();
    STATE.a_b.setZero();
    STATE.w_b.setZero();
}

TightlyCoupled::~TightlyCoupled() {}

void TightlyCoupled::setDt(const double delta_t){
    dt = delta_t;
}

void TightlyCoupled::setZ(const UwbData<double> currUwbData){
    for(int i=0; i<16; i++){
        vecZ(i) = currUwbData.distance(i);
    }
}

void TightlyCoupled::setImuVar(const double stdV, const double stdW){
    covQ.block<3,3>(9,9) = stdV*Eigen::Matrix3d::Identity();
    covQ.block<3,3>(12,12) = stdW*Eigen::Matrix3d::Identity();
}

void TightlyCoupled::setAnchorPositions(const Eigen::Matrix<double, 3, 8> &anchorpositions){
    anchorPositions = anchorpositions;
    std::cout <<"Anchor positions: \n"<<anchorPositions<<std::endl;
}

Eigen::Matrix3d TightlyCoupled::vectorToSkewSymmetric(const Eigen::Vector3d &vector){
    Eigen::Matrix3d Rot;
    Rot << 0, -vector.z(), vector.y(),
          vector.z(), 0, -vector.x(),
          -vector.y(), vector.x(), 0;
    
    return Rot;
}

Eigen::Matrix3d TightlyCoupled::Exp(const Eigen::Vector3d &omega){
    double angle = omega.norm();
    Eigen::Matrix3d Rot;
    
    if (angle<TOL){
        Rot = Eigen::Matrix3d::Identity();
    }
    else{
        Eigen::Vector3d axis = omega/angle;
        double c = cos(angle);
        double s = sin(angle);

        Rot = c*Eigen::Matrix3d::Identity() + (1 - c)*axis*axis.transpose() + s*vectorToSkewSymmetric(axis);
    } 
    return Rot;
}

void TightlyCoupled::motionModelJacobian(const ImuData<double> &imu_data){
    Eigen::Matrix3d Rot = STATE.R;
    jacobianMatF.block<3, 3>(0,3) = 0.5*dt*dt*Eigen::Matrix3d::Identity();
    jacobianMatF.block<3, 3>(0,6) = dt*Eigen::Matrix3d::Identity();
    jacobianMatF.block<3, 3>(0,9) = -0.5*dt*dt*Rot;
    jacobianMatF.block<3, 3>(3,3) = Exp((imu_data.gyr - STATE.w_b)*dt);
    jacobianMatF.block<3, 3>(3,12) = -Eigen::Matrix3d::Identity()*dt;
    jacobianMatF.block<3, 3>(6,9) = -Rot*dt;
    jacobianMatF.block<3, 3>(6,3) = -Rot*vectorToSkewSymmetric(imu_data.acc-STATE.a_b)*dt;
}

void TightlyCoupled::motionModel(const ImuData<double> &imu_data){
    Eigen::Matrix3d Rot = STATE.R;
    Eigen::Vector3d accWorld = Rot*(imu_data.acc - STATE.a_b) + _g;
    STATE.p += STATE.v*dt+0.5*accWorld*dt*dt;
    STATE.v += accWorld*dt;
    STATE.R = Rot*Exp((imu_data.gyr - STATE.w_b)*dt);
}

void TightlyCoupled::prediction(const ImuData<double> &imu_data){
    motionModelJacobian(imu_data);
    motionModel(imu_data);
    covP = jacobianMatF*covP*jacobianMatF.transpose() + covQ;
}

void TightlyCoupled::measurementModel(){
    Eigen::Vector3d p = STATE.p;
    Eigen::Vector3d offsetLeft(-0.15, 0, 0);
    Eigen::Vector3d offsetRight(0.15, 0, 0);
    for (int tag = 0; tag < 2; tag++){
        Eigen::Vector3d sensorPos = p + STATE.R * (tag == 0 ? offsetLeft : offsetRight);
        for (int i = 0; i < anchorPositions.cols(); i++){
            int idx = tag * anchorPositions.cols() + i; // 0~7: 왼쪽 tag, 8~15: 오른쪽 tag
            vecH(idx) = (sensorPos - anchorPositions.col(i)).norm();
        }
    }
}

void TightlyCoupled::measurementModelJacobian(){
    Eigen::Vector3d p = STATE.p;
    Eigen::Vector3d offset_left(-0.15, 0, 0);
    Eigen::Vector3d offset_right(0.15, 0, 0);

    // 초기화: jacobianMatH 크기는 (16 x 15)
    jacobianMatH.setZero();

    for (int tag = 0; tag < 2; tag++){
        Eigen::Vector3d offset = (tag == 0 ? offset_left : offset_right);
        Eigen::Vector3d sensorPos = p + STATE.R * offset;
        // 회전에 대한 미분은: d(sensorPos)/d(theta) = - STATE.R * skew(offset)
        Eigen::Matrix3d skew_offset = vectorToSkewSymmetric(offset);

        for (int i = 0; i < anchorPositions.cols(); i++){
            int idx = tag * anchorPositions.cols() + i;
            Eigen::Vector3d diff = sensorPos - anchorPositions.col(i);
            double norm_val = diff.norm();

            // p에 대한 편미분: diff / norm
            jacobianMatH(idx, 0) = diff(0) / norm_val;
            jacobianMatH(idx, 1) = diff(1) / norm_val;
            jacobianMatH(idx, 2) = diff(2) / norm_val;

            // 회전 (소규모 회전 업데이트를 가정하면, 센서 위치 변화는 -R*skew(offset) * delta_theta)
            // 따라서 h(x) 에 대한 회전 편미분은
            // (diff^T / norm) * (-STATE.R * skew(offset))
            Eigen::RowVector3d dtheta = (diff.transpose() / norm_val) * (-STATE.R * skew_offset);
            jacobianMatH.block<1, 3>(idx, 3) = dtheta;

            // 나머지 상태 변수(속도, 가속도 바이어스, 자이로 바이어스)에 대한 미분은 0으로 두었음.
        }
    }
}

StateforEKF<double> TightlyCoupled::correction(){
    measurementModel();
    measurementModelJacobian();
    Eigen::Matrix<double, 16, 16> residualCov;
    Eigen::Matrix<double, 15, 16> K;
    Eigen::Matrix<double, 15, 1> updateState;
    residualCov = jacobianMatH*covP*jacobianMatH.transpose() + covR;
    if (residualCov.determinant() == 0 || !residualCov.allFinite()) {
        std::cerr << "residualCov is singular or contains NaN/Inf" << std::endl;
        return STATE;
    }
    K = covP*jacobianMatH.transpose()*residualCov.inverse();
    updateState = K*(vecZ-vecH);
    STATE.p += updateState.segment<3>(0);
    STATE.R = STATE.R*Exp(updateState.segment<3>(3));
    STATE.v += updateState.segment<3>(6);
    STATE.a_b += updateState.segment<3>(9);
    STATE.w_b += updateState.segment<3>(12);
    covP = (Eigen::Matrix<double, 15, 15>::Identity()-(K*jacobianMatH))*covP;
    return STATE;
}

void TightlyCoupled::setState(StateforEKF<double> &State){
    STATE = State;
}

StateforEKF<double> TightlyCoupled::getState(){
    return STATE;
}
