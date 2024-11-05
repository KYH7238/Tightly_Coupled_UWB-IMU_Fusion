#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>

template <typename scalar>
struct ImuData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double timestamp;
    Eigen::Matrix<scalar, 3, 1> acc;
    Eigen::Matrix<scalar, 3, 1> gyr;
};

template <typename scalar>
struct UwbData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double timestamp;
    Eigen::Matrix<scalar, 8, 1> distance;
};

template <typename scalar>
struct StateforEKF{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<scalar, 3, 1> p;
    Eigen::Matrix<scalar, 3, 3> R;
    Eigen::Matrix<scalar, 3, 1> v;
    Eigen::Matrix<scalar, 3, 1> a_b;
    Eigen::Matrix<scalar, 3, 1> w_b;
};

template <typename scalar>
struct StateforESKF{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<scalar, 3, 1> p;
    Eigen::Quaterniond q;  
    Eigen::Matrix<scalar, 3, 1> v;
    Eigen::Matrix<scalar, 3, 1> a_b;
    Eigen::Matrix<scalar, 3, 1> w_b;
};
