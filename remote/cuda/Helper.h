#pragma once

#include <iostream>

#define CHECK_SUCCESS(func, SUCC, module) \
{ \
    auto status = (func); \
    if (status != SUCC) { \
        std::cerr << "Error in module [" << module << "], error num = " << status << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

#define CUDA_CHECK(func) CHECK_SUCCESS(func, cudaSuccess, "CUDA")
#define CUSOLVER_CHECK(func) CHECK_SUCCESS(func, CUSOLVER_STATUS_SUCCESS, "cuSolver")
#define CUBLAS_CHECK(func) CHECK_SUCCESS(func, CUBLAS_STATUS_SUCCESS, "cuBLAS")