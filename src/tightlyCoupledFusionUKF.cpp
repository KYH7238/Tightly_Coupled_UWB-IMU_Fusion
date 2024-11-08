/**
 * Author: Yonghee Kim
 * Date: 2024-11-5
 * brief: 6D pose estimation using tightly coupled UWB/IMU fusion using UKF
 */
#include "tightlyCoupledFusionUKF.h" 
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm> 

TightlyCoupled::TightlyCoupled() {
    covP = Eigen::Matrix<double, 15, 15>::Zero();
    covQ = Eigen::Matrix<double, 15, 15>::Identity() * 0.001;
    covR = Eigen::Matrix<double, 8, 8>::Identity() * 0.08;

    anchorPositions = Eigen::Matrix<double, 3, 8>::Zero();
    _g << 0, 0, 9.81;
    TOL = 1e-9;
    dt = 0;

    STATE.p = Eigen::Vector3d::Zero();
    STATE.R = Eigen::Matrix3d::Identity();
    STATE.v = Eigen::Vector3d::Zero();
    STATE.a_b = Eigen::Vector3d::Zero();
    STATE.w_b = Eigen::Vector3d::Zero();

    int n = 15; 
    double alpha = 1e-3;
    double beta = 2.0;
    double kappa = 0.0;
    lambda = alpha * alpha * (n + kappa) - n;

    weightsMean = Eigen::VectorXd::Constant(2 * n + 1, 0.5 / (n + lambda));
    weightsCov = weightsMean;
    weightsMean(0) = lambda / (n + lambda);
    weightsCov(0) = weightsMean(0) + (1 - alpha * alpha + beta);
}

TightlyCoupled::~TightlyCoupled() {}

void TightlyCoupled::setDt(const double delta_t) {
    dt = delta_t;
}

void TightlyCoupled::setImuVar(const double stdV, const double stdW) {
    covQ.block<3,3>(9,9) = stdV * Eigen::Matrix3d::Identity();
    covQ.block<3,3>(12,12) = stdW * Eigen::Matrix3d::Identity();
}

void TightlyCoupled::setAnchorPositions(const Eigen::Matrix<double, 3, 8> &anchorpositions) {
    anchorPositions = anchorpositions;
    std::cout << "Anchor positions: \n" << anchorPositions << std::endl;
}

Eigen::Matrix3d TightlyCoupled::vectorToSkewSymmetric(const Eigen::Vector3d &vector) {
    Eigen::Matrix3d skewSymmetric;
    skewSymmetric <<    0,        -vector(2),  vector(1),
                     vector(2),       0,      -vector(0),
                    -vector(1),  vector(0),        0;
    return skewSymmetric;
}

Eigen::Matrix3d TightlyCoupled::Exp(const Eigen::Vector3d &omega) {
    double angle = omega.norm();
    Eigen::Matrix3d Rot;
    if (angle < TOL) {
        Rot = Eigen::Matrix3d::Identity();
    } else {
        Eigen::Vector3d axis = omega / angle;
        double c = cos(angle);
        double s = sin(angle);
        Rot = c * Eigen::Matrix3d::Identity() + (1 - c) * axis * axis.transpose() + s * vectorToSkewSymmetric(axis);
    }
    return Rot;
}

Eigen::Vector3d TightlyCoupled::vee(const Eigen::Matrix3d &phi) {
    Eigen::Vector3d v;
    v << phi(2, 1), phi(0, 2), phi(1, 0);
    return v;
}

Eigen::Vector3d TightlyCoupled::Log(const Eigen::Matrix3d &Rot) {
    double trace = Rot.trace();
    double cos_angle = (trace - 1.0) / 2.0;
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle)); 
    double angle = acos(cos_angle);
    Eigen::Vector3d phi;

    if (std::abs(angle) < TOL) {
        phi = vee(Rot - Eigen::Matrix3d::Identity());
    } else {
        phi = vee((0.5 * angle / sin(angle)) * (Rot - Rot.transpose()));
    }
    return phi;
}

Eigen::VectorXd TightlyCoupled::phiInv(const StateforEKF<double> &state, const StateforEKF<double> &hatState) {
    Eigen::VectorXd xi(15);
    xi.segment<3>(0) = hatState.p - state.p;
    xi.segment<3>(3) = Log(hatState.R * state.R.transpose());
    xi.segment<3>(6) = hatState.v - state.v;
    xi.segment<3>(9) = hatState.a_b - state.a_b;
    xi.segment<3>(12) = hatState.w_b - state.w_b;
    return xi;
}

StateforEKF<double> TightlyCoupled::phi(const StateforEKF<double> &state, const Eigen::VectorXd &xi) {
    StateforEKF<double> newState;
    newState.p = state.p + xi.segment<3>(0);
    newState.R = Exp(xi.segment<3>(3)) * state.R;
    newState.v = state.v + xi.segment<3>(6);
    newState.a_b = state.a_b + xi.segment<3>(9);
    newState.w_b = state.w_b + xi.segment<3>(12);
    return newState;
}

Eigen::VectorXd TightlyCoupled::h(const StateforEKF<double> &state) {
    Eigen::VectorXd ranges(8);
    for (int i = 0; i < 8; ++i) {
        ranges(i) = (state.p - anchorPositions.col(i)).norm();
    }
    return ranges;
}

void TightlyCoupled::generateSigmaPoints() {
    int n = 15; 
    sigmaPoints.clear();
    Eigen::VectorXd x(15);
    x.segment<3>(0) = STATE.p;
    x.segment<3>(3) = Log(STATE.R);
    x.segment<3>(6) = STATE.v;
    x.segment<3>(9) = STATE.a_b;
    x.segment<3>(12) = STATE.w_b;

    Eigen::MatrixXd sqrtP = covP.llt().matrixL();

    sigmaPoints.push_back(STATE); 

    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd delta = sqrt(lambda + n) * sqrtP.col(i);
        StateforEKF<double> sigmaStatePlus = phi(STATE, delta);
        StateforEKF<double> sigmaStateMinus = phi(STATE, -delta);
        sigmaPoints.push_back(sigmaStatePlus);
        sigmaPoints.push_back(sigmaStateMinus);
    }
}

void TightlyCoupled::predictSigmaPoints(const ImuData<double> &imu_data) {
    predictedSigmaPoints.clear();
    for (auto& sigmaState : sigmaPoints) {
        StateforEKF<double> predictedState;
        Eigen::Vector3d accWorld = sigmaState.R * (imu_data.acc - sigmaState.a_b) + _g;
        predictedState.p = sigmaState.p + sigmaState.v * dt + 0.5 * accWorld * dt * dt;
        predictedState.v = sigmaState.v + accWorld * dt;
        Eigen::Matrix3d dR = Exp((imu_data.gyr - sigmaState.w_b) * dt);
        predictedState.R = sigmaState.R * dR;
        predictedState.a_b = sigmaState.a_b;
        predictedState.w_b = sigmaState.w_b;
        predictedSigmaPoints.push_back(predictedState);
    }
}

void TightlyCoupled::predictMeanAndCovariance() {
    int n = 15;
    Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(n);
    for (size_t i = 0; i < predictedSigmaPoints.size(); ++i) {
        Eigen::VectorXd xi = phiInv(predictedSigmaPoints[0], predictedSigmaPoints[i]);
        x_pred += weightsMean(i) * xi;
    }
    STATE = phi(predictedSigmaPoints[0], x_pred);

    covP = Eigen::MatrixXd::Zero(n, n);
    for (size_t i = 0; i < predictedSigmaPoints.size(); ++i) {
        Eigen::VectorXd xi = phiInv(STATE, predictedSigmaPoints[i]);
        covP += weightsCov(i) * xi * xi.transpose();
    }
    covP += covQ; 
}

void TightlyCoupled::predictMeasurement() {
    int m = 8; 
    predictedMeasurements.clear();
    for (auto& sigmaState : predictedSigmaPoints) {
        Eigen::VectorXd z = h(sigmaState);
        predictedMeasurements.push_back(z);
    }

    z_pred = Eigen::VectorXd::Zero(m);
    for (size_t i = 0; i < predictedMeasurements.size(); ++i) {
        z_pred += weightsMean(i) * predictedMeasurements[i];
    }

    S = Eigen::MatrixXd::Zero(m, m);
    for (size_t i = 0; i < predictedMeasurements.size(); ++i) {
        Eigen::VectorXd z_diff = predictedMeasurements[i] - z_pred;
        S += weightsCov(i) * z_diff * z_diff.transpose();
    }
    S += covR; 

    int n = 15;
    Tc = Eigen::MatrixXd::Zero(n, m);
    for (size_t i = 0; i < predictedSigmaPoints.size(); ++i) {
        Eigen::VectorXd x_diff = phiInv(STATE, predictedSigmaPoints[i]);
        Eigen::VectorXd z_diff = predictedMeasurements[i] - z_pred;
        Tc += weightsCov(i) * x_diff * z_diff.transpose();
    }
}

void TightlyCoupled::updateState(const UwbData<double>& uwb_data) {
    Eigen::MatrixXd K = Tc * S.inverse();
    Eigen::VectorXd z = uwb_data.distance;
    Eigen::VectorXd z_diff = z - z_pred;
    Eigen::VectorXd x_update = K * z_diff;
    STATE = phi(STATE, x_update);
    covP = covP - K * S * K.transpose();
}

void TightlyCoupled::prediction(const ImuData<double> &imu_data) {
    generateSigmaPoints();
    predictSigmaPoints(imu_data);
    predictMeanAndCovariance();
}

void TightlyCoupled::update(const UwbData<double>& uwb_data) {
    predictMeasurement();
    updateState(uwb_data);
}

StateforEKF<double> TightlyCoupled::getState() {
    return STATE;
}

void TightlyCoupled::setState(const StateforEKF<double> &State) {
    STATE = State;
}