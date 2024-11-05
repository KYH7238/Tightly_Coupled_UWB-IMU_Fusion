/**
 * Author: Yonghee Kim
 * Date: 2024-11-5
 * brief: 6D pose estimation using tightly coupled UWB/IMU fusion using filtering method(EKF/ESKF/UKF/LIEKF)
 */
#include "tightlyCoupledFusionEKF.h" 
#include <Eigen/Dense>
#include <Eigen/Geometry>

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
    for(int i=0; i<8; i++){
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
    for (int i=0; i<vecH.size(); i++){
        vecH(i) = (p - anchorPositions.col(i)).norm();
    }
}

void TightlyCoupled::measurementModelJacobian(){
    Eigen::Vector3d p = STATE.p;
    for (int i=0; i<jacobianMatH.rows(); i++){
        jacobianMatH(i,0) = (p(0)-anchorPositions(0,i))/vecH(i);
        jacobianMatH(i,1) = (p(1)-anchorPositions(1,i))/vecH(i);
        jacobianMatH(i,2) = (p(2)-anchorPositions(2,i))/vecH(i); 
    }
}

StateforEKF<double> TightlyCoupled::correction(){
    measurementModel();
    measurementModelJacobian();
    Eigen::Matrix<double, 8, 8> residualCov;
    Eigen::Matrix<double, 15, 8> K;
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
