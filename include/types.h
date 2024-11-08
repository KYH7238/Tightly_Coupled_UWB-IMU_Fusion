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

struct WEIGHTS {
    struct W {
        double sqrtDlambda;
        double wj;
        double wm;
        double w0;

        W(double l, double alpha) {
            double m = (std::pow(alpha, 2) - 1) * l;
            sqrtDlambda = std::sqrt(l + m);
            wj = 1 / (2 * (l + m));
            wm = m / (l + m);
            w0 = m / (l + m) + 3 - std::pow(alpha, 2);
        }
    };
    
    W redD;
    W q;
    W upD;

    WEIGHTS(double red_d_val, double q_val, double up_d_val, double* alpha) :
    //             15                12            3        diag[0.001, 0.001, 0.001]   
    redD(red_d_val, alpha[0]),
    q(q_val, alpha[1]),
    upD(up_d_val, alpha[2])
 {}
};
