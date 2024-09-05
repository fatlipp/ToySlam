#pragma once

#include <vector>
#include <cstdint>

std::vector<uint8_t> UintToBytes(uint32_t value);

template<typename T>
std::vector<uint8_t> FloatToBytes(T value);

template<typename T>
std::vector<uint8_t> MatrixToByteArray(const int rows, const int cols, const T* matrix, bool is_diag);