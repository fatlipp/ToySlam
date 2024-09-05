#pragma once

#include <Eigen/Eigen>
#include <iostream>

template<typename T>
static inline Eigen::Matrix3<T> CreateTransform2d(const T x, const T y, const T theta)
{
    const Eigen::AngleAxis<T> rollAngle(theta, Eigen::Vector3<T>::UnitZ());
    const Eigen::AngleAxis<T> yawAngle(0, Eigen::Vector3<T>::UnitY());
    const Eigen::AngleAxis<T> pitchAngle(0, Eigen::Vector3<T>::UnitX());
    const Eigen::Quaternion<T> q = rollAngle * yawAngle * pitchAngle;
    Eigen::Matrix3<T> mat = q.matrix();
    mat(0, 2) = x;
    mat(1, 2) = y;

    return mat;
}

template<typename T>
static inline T ConvertMatToAngle(Eigen::Matrix3<T> R)
{  
    return std::atan2(R(1, 0), R(0, 0));
}

template<typename T>
static inline Eigen::Matrix3<T> InverseTransform2d(const Eigen::Matrix3<T>& pos)
{
    Eigen::Transform<T, 2, Eigen::TransformTraits::Affine> posAffine(pos);
    return posAffine.inverse(Eigen::TransformTraits::Affine).matrix();
}