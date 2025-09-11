/**
 * Author: Yonghee Kim
 * Date: 2025-09-11
 * brief: 6D pose estimation using loosely coupled UWB/IMU fusion using filtering method(EKF/ESKF/UKF/LIEKF)
 */
#include "looselyCoupledFusionUKF.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>

LooselyCoupled::LooselyCoupled() {
    covP = Eigen::Matrix<double, 15, 15>::Zero();
    covQ = Eigen::Matrix<double, 15, 15>::Identity() * 0.001;
    covR = Eigen::Matrix<double, 3, 3>::Identity() * 0.08;   

    _g << 0, 0, 9.81;
    TOL = 1e-9;
    dt = 0;

    STATE.p.setZero();
    STATE.R.setIdentity();
    STATE.v.setZero();
    STATE.a_b.setZero();
    STATE.w_b.setZero();

    const int n = 15;
    double alpha = 1e-3;
    double beta  = 2.0;
    double kappa = 0.0;
    lambda = alpha * alpha * (n + kappa) - n;

    weightsMean = Eigen::VectorXd::Constant(2 * n + 1, 0.5 / (n + lambda));
    weightsCov  = weightsMean;
    weightsMean(0) = lambda / (n + lambda);
    weightsCov(0)  = weightsMean(0) + (1 - alpha * alpha + beta);
}

LooselyCoupled::~LooselyCoupled() {}

void LooselyCoupled::setDt(const double delta_t) {
    dt = delta_t;
}

void LooselyCoupled::setImuVar(const double stdV, const double stdW) {
    covQ.block<3,3>(9,9)   = stdV * Eigen::Matrix3d::Identity();
    covQ.block<3,3>(12,12) = stdW * Eigen::Matrix3d::Identity();
}

Eigen::Matrix3d LooselyCoupled::vectorToSkewSymmetric(const Eigen::Vector3d &v) {
    Eigen::Matrix3d S;
    S <<     0, -v.z(),  v.y(),
          v.z(),     0, -v.x(),
         -v.y(),  v.x(),     0;
    return S;
}

Eigen::Matrix3d LooselyCoupled::Exp(const Eigen::Vector3d &omega) {
    double angle = omega.norm();
    if (angle < TOL) return Eigen::Matrix3d::Identity();
    Eigen::Vector3d axis = omega / angle;
    double c = std::cos(angle), s = std::sin(angle);
    return c * Eigen::Matrix3d::Identity()
         + (1 - c) * axis * axis.transpose()
         + s * vectorToSkewSymmetric(axis);
}

Eigen::Vector3d LooselyCoupled::vee(const Eigen::Matrix3d &phi) {
    return Eigen::Vector3d(phi(2,1), phi(0,2), phi(1,0));
}

Eigen::Vector3d LooselyCoupled::Log(const Eigen::Matrix3d &R) {
    double tr = R.trace();
    double c = std::max(-1.0, std::min(1.0, (tr - 1.0) / 2.0));
    double ang = std::acos(c);
    if (std::abs(ang) < TOL) return vee(R - Eigen::Matrix3d::Identity());
    return vee((0.5 * ang / std::sin(ang)) * (R - R.transpose()));
}

Eigen::VectorXd LooselyCoupled::phiInv(const StateforEKF<double> &x, const StateforEKF<double> &xh) {
    Eigen::VectorXd xi(15);
    xi.segment<3>(0) = xh.p - x.p;
    xi.segment<3>(3) = Log(xh.R * x.R.transpose());
    xi.segment<3>(6) = xh.v - x.v;
    xi.segment<3>(9) = xh.a_b - x.a_b;
    xi.segment<3>(12)= xh.w_b - x.w_b;
    return xi;
}

StateforEKF<double> LooselyCoupled::phi(const StateforEKF<double> &x, const Eigen::VectorXd &xi) {
    StateforEKF<double> y = x;
    y.p   = x.p + xi.segment<3>(0);
    y.R   = Exp(xi.segment<3>(3)) * x.R;
    y.v   = x.v + xi.segment<3>(6);
    y.a_b = x.a_b + xi.segment<3>(9);
    y.w_b = x.w_b + xi.segment<3>(12);
    return y;
}

Eigen::VectorXd LooselyCoupled::h(const StateforEKF<double> &state) {
    Eigen::VectorXd z(3);
    z = state.p; 
    return z;
}

void LooselyCoupled::generateSigmaPoints() {
    const int n = 15;
    sigmaPoints.clear();

    Eigen::VectorXd x(15);
    x.segment<3>(0)  = STATE.p;
    x.segment<3>(3)  = Log(STATE.R);
    x.segment<3>(6)  = STATE.v;
    x.segment<3>(9)  = STATE.a_b;
    x.segment<3>(12) = STATE.w_b;

    Eigen::MatrixXd sqrtP = covP.llt().matrixL();
    sigmaPoints.push_back(STATE); 

    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd delta = std::sqrt(lambda + n) * sqrtP.col(i);
        StateforEKF<double> sp = phi(STATE,  delta);
        StateforEKF<double> sm = phi(STATE, -delta);
        sigmaPoints.push_back(sp);
        sigmaPoints.push_back(sm);
    }
}

void LooselyCoupled::predictSigmaPoints(const ImuData<double> &imu_data) {
    predictedSigmaPoints.clear();
    for (const auto& s : sigmaPoints) {
        StateforEKF<double> y;
        Eigen::Vector3d accW = s.R * (imu_data.acc - s.a_b) + _g;
        y.p   = s.p + s.v * dt + 0.5 * accW * dt * dt;
        y.v   = s.v + accW * dt;
        y.R   = s.R * Exp((imu_data.gyr - s.w_b) * dt);
        y.a_b = s.a_b;
        y.w_b = s.w_b;
        predictedSigmaPoints.push_back(y);
    }
}

void LooselyCoupled::predictMeanAndCovariance() {
    const int n = 15;
    Eigen::VectorXd xpred = Eigen::VectorXd::Zero(n);
    for (size_t i = 0; i < predictedSigmaPoints.size(); ++i) {
        Eigen::VectorXd xi = phiInv(predictedSigmaPoints[0], predictedSigmaPoints[i]);
        xpred += weightsMean(i) * xi;
    }
    STATE = phi(predictedSigmaPoints[0], xpred);

    covP = Eigen::MatrixXd::Zero(n, n);
    for (size_t i = 0; i < predictedSigmaPoints.size(); ++i) {
        Eigen::VectorXd xi = phiInv(STATE, predictedSigmaPoints[i]);
        covP += weightsCov(i) * xi * xi.transpose();
    }
    covP += covQ;
}

void LooselyCoupled::predictMeasurement() {
    const int m = 3;  
    predictedMeasurements.clear();
    predictedMeasurements.reserve(predictedSigmaPoints.size());

    for (const auto& sp : predictedSigmaPoints) {
        predictedMeasurements.push_back(h(sp));
    }

    z_pred = Eigen::VectorXd::Zero(m);
    for (size_t i = 0; i < predictedMeasurements.size(); ++i) {
        z_pred += weightsMean(i) * predictedMeasurements[i];
    }

    S  = Eigen::MatrixXd::Zero(m, m);
    Tc = Eigen::MatrixXd::Zero(15, m);
    for (size_t i = 0; i < predictedMeasurements.size(); ++i) {
        Eigen::VectorXd z_diff = predictedMeasurements[i] - z_pred;
        S  += weightsCov(i) * z_diff * z_diff.transpose();

        Eigen::VectorXd x_diff = phiInv(STATE, predictedSigmaPoints[i]);
        Tc += weightsCov(i) * x_diff * z_diff.transpose();
    }
    S += covR;
}

void LooselyCoupled::updateState(const UwbData<double>& uwb_data) {
    Eigen::MatrixXd K = Tc * S.inverse();
    Eigen::VectorXd z(3);
    z << uwb_data.pos(0), uwb_data.pos(1), uwb_data.pos(2);

    Eigen::VectorXd z_diff = z - z_pred;
    Eigen::VectorXd x_upd  = K * z_diff;

    STATE = phi(STATE, x_upd);
    covP  = covP - K * S * K.transpose();
}

void LooselyCoupled::prediction(const ImuData<double> &imu_data) {
    generateSigmaPoints();
    predictSigmaPoints(imu_data);
    predictMeanAndCovariance();
}

void LooselyCoupled::update(const UwbData<double>& uwb_data) {
    predictMeasurement();
    updateState(uwb_data);
}

StateforEKF<double> LooselyCoupled::getState() {
    return STATE;
}

void LooselyCoupled::setState(const StateforEKF<double> &State) {
    STATE = State;
}
