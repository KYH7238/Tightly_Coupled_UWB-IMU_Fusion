#pragma once
#include "types.h"
#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include <iostream>
#include <vector>

class TightlyCoupled {
public:
    TightlyCoupled();
    ~TightlyCoupled();

    void setImuVar(const double stdV, const double stdW);
    void setDt(const double delta_t);
    void setAnchorPositions(const Eigen::Matrix<double, 3, 8> &anchorPositions);
    void setZ(const UwbData<double>& currUwbData);    

    void prediction(const ImuData<double> &imu_data);
    void update(const UwbData<double>& uwb_data);

    StateforEKF<double> getState();
    void setState(const StateforEKF<double> &State);

private:
    Eigen::Matrix3d vectorToSkewSymmetric(const Eigen::Vector3d &vector);
    Eigen::Matrix3d Exp(const Eigen::Vector3d &omega);
    Eigen::Vector3d Log(const Eigen::Matrix3d &Rot);
    Eigen::Vector3d vee(const Eigen::Matrix3d &phi);

    Eigen::VectorXd phiInv(const StateforEKF<double> &state, const StateforEKF<double> &hatState);
    StateforEKF<double> phi(const StateforEKF<double> &state, const Eigen::VectorXd &xi);
    Eigen::VectorXd h(const StateforEKF<double> &state);

    void generateSigmaPoints();
    void predictSigmaPoints(const ImuData<double> &imu_data);
    void predictMeanAndCovariance();
    void predictMeasurement();
    void updateState(const UwbData<double>& uwb_data);

    Eigen::Matrix<double, 15, 15> covP;
    Eigen::Matrix<double, 15, 15> covQ;
    Eigen::Matrix<double, 8, 8> covR;

    Eigen::Matrix<double, 3, 8> anchorPositions;
    Eigen::VectorXd weightsMean;
    Eigen::VectorXd weightsCov;
    double lambda;

    double dt;
    double TOL;
    Eigen::Vector3d _g;
    StateforEKF<double> STATE;
    std::vector<StateforEKF<double>> sigmaPoints;
    std::vector<StateforEKF<double>> predictedSigmaPoints;
    std::vector<Eigen::VectorXd> predictedMeasurements;

    Eigen::VectorXd z_pred;
    Eigen::MatrixXd S;
    Eigen::MatrixXd Tc;
};


// #pragma once
// #include "types.h"
// #include <Eigen/Dense>
// #include <Eigen/Geometry> 
// #include <iostream>
// #include <vector>

// class TightlyCoupled {
// public:
//     TightlyCoupled();
//     ~TightlyCoupled();
//     void setImuVar(const double stdV, const double stdW);
//     void setDt(const double delta_t);
//     void setAnchorPositions(const Eigen::Matrix<double, 3, 8> &anchorPositions);
//     Eigen::Matrix3d Exp(const Eigen::Vector3d &omega);
//     void prediction(const ImuData<double> &imu_data);
//     void setZ(const UwbData<double> currUwbData);    
//     void motionModel(const ImuData<double> &imu_data);
//     void motionModelJacobian(const ImuData<double> &imu_data);
//     Eigen::Matrix3d vectorToSkewSymmetric(const Eigen::Vector3d &vector);
//     void measurementModel();
//     void measurementModelJacobian ();
//     StateforEKF<double> correction();
//     void setState(StateforEKF<double> &State);
//     StateforEKF<double> getState();

// private:
//     Eigen::Matrix<double, 15, 15> covP, jacobianMatF;
//     Eigen::Matrix<double, 12, 12> covQ;
//     Eigen::Matrix<double, 8, 8> covR;
//     Eigen::Vector<double, 3> upIdx;
//     Eigen::Matrix<double, 12, 12> cholQ;
//     Eigen::Vector<double,15> redIdxs;

//     Eigen::Matrix<double, 8, 1> vecZ, vecH;
//     Eigen::Matrix<double, 3, 8> anchorPositions;
//     Eigen::Matrix<double, 8, 15> jacobianMatH;
//     double dt, TOL;
//     Eigen::Matrix<double, 3, 1> _g;
//     StateforEKF<double> STATE;

// };