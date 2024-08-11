#pragma once

#include <Eigen/Eigen>

static Eigen::Matrix3f CreateTransform2d(const float x, const float y, const float theta)
{
    Eigen::AngleAxisf rollAngle(theta, Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yawAngle(0, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitchAngle(0, Eigen::Vector3f::UnitX());
    Eigen::Quaternion<float> q = rollAngle * yawAngle * pitchAngle;
    Eigen::Matrix3f mat = q.matrix();
    mat(0, 2) = x;
    mat(1, 2) = y;

    return mat;
}

static float ConvertMatToAngle(Eigen::Matrix3f R)
{  
    return std::atan2(R(1, 0), R(0, 0));
}