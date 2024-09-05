get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_DIR "${CUDA_COMPILER_BIN}" DIRECTORY)
set(CUDA_INCLUDE "${CUDA_DIR}/include")

find_package(Eigen3 REQUIRED)

set(PROJ_NAME "cuda_core")
set(SOURCE
    solver/SolverCudaOps.cpp
    optimizer/kernels/OptimizerGpu.cu
    optimizer/kernels/KernelSe2.cu
    optimizer/kernels/KernelSe2Point2.cu
    optimizer/kernels/KernelCommon.cu
)

add_library(${PROJ_NAME} STATIC)
target_sources(${PROJ_NAME}
    PRIVATE
        ${SOURCE}
)
target_include_directories(${PROJ_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_REMOTE_DIR}
        ${CUDA_INCLUDE}
)
target_link_libraries(${PROJ_NAME} 
    PRIVATE
        cublas
        cudart
        cusolver
        Eigen3::Eigen
    )
set_property(TARGET ${PROJ_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_options(${PROJ_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(${PROJ_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJ_NAME} PROPERTIES CUDA_ARCHITECTURES "86")