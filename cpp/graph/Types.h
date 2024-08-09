#pragma once

#include <Eigen/Eigen>

template<typename T>
using DynamicMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template<typename T>
using DynamicVector = Eigen::Vector<T, Eigen::Dynamic>;