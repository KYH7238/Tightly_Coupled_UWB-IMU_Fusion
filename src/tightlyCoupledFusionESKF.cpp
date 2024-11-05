/**
 * Author: Yonghee Kim
 * Date: 2024-11-5
 * brief: 6D pose estimation using tightly coupled UWB/IMU fusion using filtering method(EKF/ESKF/UKF/LIEKF)
 */
#include "tightlyCoupledFusionESKF.h" 
#include <Eigen/Dense>
#include <Eigen/Geometry>


TightlyCoupled::TightlyCoupled()
{
    covP.setZero();
    covQ.setIdentity()*0.001;
    covR.setIdentity()*0.08;
    vecZ.setZero();
    vecH.setZero();
    XdX.setZero();
    jacobianMatF.setIdentity();
    jacobianMatH.setZero();
    anchorPositions.setZero();
    _g << 0, 0, 9.81;

    TOL =1e-9;
    dt = 0;
    STATE.p <<0, 0, 0;
    STATE.q = Eigen::Quaterniond::Identity();

    STATE.v.setZero();
    STATE.a_b.setZero();
    STATE.w_b.setZero();
    updateH();
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

Eigen::Quaterniond TightlyCoupled::getQuaFromAA(Eigen::Matrix<double, 3, 1> vec){
    double theta = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    if (0 == theta)
        return Eigen::Quaterniond::Identity();
    Eigen::Matrix<double, 3, 1> unitAxis = vec / theta;
    double w = cos(0.5 * theta);
    double sinT = sin(0.5 * theta);
    double x = unitAxis[0] * sinT;
    double y = unitAxis[1] * sinT;
    double z = unitAxis[2] * sinT;
    Eigen::Quaterniond qua(w, x, y, z);
    qua.normalized();

    return qua;
}

void TightlyCoupled::motionModelJacobian(const ImuData<double> &imu_data){
    Eigen::Matrix3d Rot = STATE.q.toRotationMatrix();
    jacobianMatF.block<3, 3>(0,6) = dt * Eigen::Matrix3d::Identity();
    jacobianMatF.block<3, 3>(6,9) = -Rot*dt;
    jacobianMatF.block<3, 3>(3,3) = Rot.transpose() * Exp((imu_data.gyr - STATE.w_b)*dt);
    jacobianMatF.block<3, 3>(3,12) = -dt * Eigen::Matrix3d::Identity();
    jacobianMatF.block<3, 3>(6,3) = -Rot*vectorToSkewSymmetric(imu_data.acc-STATE.a_b)*dt;
}

void TightlyCoupled::motionModel(const ImuData<double> &imu_data){
    Eigen::Matrix3d Rot = STATE.q.toRotationMatrix();
    Eigen::Vector3d accWorld = Rot*(imu_data.acc - STATE.a_b) + _g;
    STATE.p += STATE.v*dt+0.5*accWorld*dt*dt;
    STATE.v += accWorld*dt;
    Eigen::Quaterniond deltaQ = getQuaFromAA((imu_data.gyr - STATE.w_b)*dt);
    STATE.q = STATE.q * deltaQ;
}

void TightlyCoupled::prediction(const ImuData<double> &imu_data){
    motionModelJacobian(imu_data);
    motionModel(imu_data);
    covP = jacobianMatF*covP*jacobianMatF.transpose() + covQ;
}

void TightlyCoupled::updateH(){
    Eigen::Quaterniond quat = STATE.q;
    XdX.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    XdX.block<3,3>(7,6) = Eigen::Matrix3d::Identity();
    XdX.block<3,3>(10,9) = Eigen::Matrix3d::Identity();
    XdX.block<3,3>(13,12) = Eigen::Matrix3d::Identity();
    XdX.block<4,3>(3,3) = 0.5 * (Eigen::Matrix<double,4,3>() << 
                        -quat.x(), -quat.y(), -quat.z(),
                        quat.w(), -quat.z(),  quat.y(),
                        quat.z(),  quat.w(), -quat.x(),
                        -quat.y(),  quat.x(),  quat.w()).finished();                        

}

void TightlyCoupled::measurementModel(){
    Eigen::Vector3d p = STATE.p;
    for (int i=0; i<vecH.size(); i++){
        vecH(i) = (p - anchorPositions.col(i)).norm();
    }
}

void TightlyCoupled::measurementModelJacobian(){
    Eigen::Vector3d p = STATE.p;
    Eigen::Matrix<double, 8, 16> H;
    for (int i=0; i<jacobianMatH.rows(); i++){
        H(i,0) = (p(0)-anchorPositions(0,i))/vecH(i);
        H(i,1) = (p(1)-anchorPositions(1,i))/vecH(i);
        H(i,2) = (p(2)-anchorPositions(2,i))/vecH(i); 
    }
    updateH();
    jacobianMatH = H*XdX;
}

StateforESKF<double> TightlyCoupled::correction(){
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
    Eigen::Quaterniond deltaQ = getQuaFromAA(updateState.segment<3>(3));
    STATE.q = STATE.q*deltaQ;
    STATE.v += updateState.segment<3>(6);
    STATE.a_b += updateState.segment<3>(9);
    STATE.w_b += updateState.segment<3>(12);
    Eigen::Matrix<double, 15, 15> IKH = (Eigen::Matrix<double, 15, 15>::Identity()-(K*jacobianMatH));
    covP = IKH*covP*IKH.transpose() + K*covR*K.transpose();

    return STATE;
}
void TightlyCoupled::setState(StateforESKF<double> &State){
    STATE = State;
}
StateforESKF<double> TightlyCoupled::getState(){
    return STATE;
}
