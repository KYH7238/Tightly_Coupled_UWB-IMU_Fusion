#pragma once
#include "types.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>

class LooselyCoupled {
public:
    LooselyCoupled();
    ~LooselyCoupled();

    void setImuVar(const double stdV, const double stdW);
    void setDt(const double delta_t);
    void setZ(const UwbData<double> &uwbData);
    void prediction(const ImuData<double> &imu_data);
    void motionModel(const ImuData<double> &imu_data);
    void motionModelJacobian(const ImuData<double> &imu_data);
    Eigen::Matrix3d Exp(const Eigen::Vector3d &omega);
    Eigen::Quaterniond getQuaFromAA(const Eigen::Vector3d &vec);
    Eigen::Matrix3d vectorToSkewSymmetric(const Eigen::Vector3d &v);

    void measurementModel();
    void measurementModelJacobian();
    StateforESKF<double> correction();

    void setState(StateforESKF<double> &State);
    StateforESKF<double> getState();

private:
    Eigen::Matrix<double,15,15> covP, covQ, jacobianMatF;
    Eigen::MatrixXd covR;
    Eigen::VectorXd vecZ, vecH;
    Eigen::MatrixXd jacobianMatH;
    double dt, TOL;
    Eigen::Vector3d _g;
    int meas_dim;
    StateforESKF<double> STATE;
};
