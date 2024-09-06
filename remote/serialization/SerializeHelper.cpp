#include "serialization/SerializeHelper.h"

#include <cstring>

std::vector<uint8_t> UintToBytes(uint32_t value)
{
    std::vector<uint8_t> bytes(4);
    std::memcpy(bytes.data(), &value, sizeof(uint32_t));
    return bytes;
}


template<typename T>
std::vector<uint8_t> FloatToBytes(T value)
{
    std::vector<uint8_t> bytes(sizeof(T));
    std::memcpy(bytes.data(), &value, sizeof(T));
    return bytes;
}


template<typename T>
std::vector<uint8_t> MatrixToByteArray(const int rows, const int cols, const T* matrix, bool is_diag)
{
    std::vector<uint8_t> bytes;
    
    if (is_diag) 
    {
        const auto bytesRows = UintToBytes(0);
        bytes.insert(bytes.end(), bytesRows.begin(), bytesRows.end());
        const auto bytesCols = UintToBytes(rows);
        bytes.insert(bytes.end(), bytesCols.begin(), bytesCols.end());
        for (int i = 0; i < rows; ++i) 
        {
            std::vector<uint8_t> diagBytes = FloatToBytes<T>(matrix[(i + i * cols)]);
            bytes.insert(bytes.end(), diagBytes.begin(), diagBytes.end());
        }
    } 
    else 
    {
        const auto bytesRows = UintToBytes(rows);
        bytes.insert(bytes.end(), bytesRows.begin(), bytesRows.end());
        const auto bytesCols = UintToBytes(cols);
        bytes.insert(bytes.end(), bytesCols.begin(), bytesCols.end());
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                std::vector<uint8_t> elemBytes = FloatToBytes<T>(matrix[j + i * cols]);
                bytes.insert(bytes.end(), elemBytes.begin(), elemBytes.end());
            }
        }
    }

    return bytes;
}

template std::vector<uint8_t> FloatToBytes(float value);
template std::vector<uint8_t> FloatToBytes(double value);
template std::vector<uint8_t> MatrixToByteArray<float>(const int rows, const int cols, const float* matrix, bool is_diag);
template std::vector<uint8_t> MatrixToByteArray<double>(const int rows, const int cols, const double* matrix, bool is_diag);