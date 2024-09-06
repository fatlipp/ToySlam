#include "cuda/solver/SolverCudaOps.h"

template<>
SolverCudaOps<float>::QrFactorizationBufferSizeType SolverCudaOps<float>::QrFactorizationBufferSizeCalc = cusolverDnSgeqrf_bufferSize;
template<>
SolverCudaOps<float>::QrFactorizationType SolverCudaOps<float>::QrFactorization = cusolverDnSgeqrf;
template<>
SolverCudaOps<float>::QbyVectorBType SolverCudaOps<float>::QbyVectorB = cusolverDnSormqr;

template<>
SolverCudaOps<double>::QrFactorizationBufferSizeType SolverCudaOps<double>::QrFactorizationBufferSizeCalc = cusolverDnDgeqrf_bufferSize;
template<>
SolverCudaOps<double>::QrFactorizationType SolverCudaOps<double>::QrFactorization = cusolverDnDgeqrf;
template<>
SolverCudaOps<double>::QbyVectorBType SolverCudaOps<double>::QbyVectorB = cusolverDnDormqr;
