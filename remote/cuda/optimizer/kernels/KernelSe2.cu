#include "cuda/optimizer/kernels/KernelHelper.cuh"
#include "cuda/graph/VertexStorageGpu.h"
#include "cuda/graph/BaseVertexGpu.h"
#include "cuda/graph/BaseEdgeGpu.h"

template<typename T>
__device__ void ProcessEdgeSe2(const VertexStorageGpu<T>* vertices, const unsigned id1, const unsigned id2,
                            const T* meas, T* error)
{
    const unsigned vertexOffset1 = vertices->offsets[id1];
    const unsigned vertexOffset2 = vertices->offsets[id2];
    const T* pos_1 = reinterpret_cast<VertexGpuSe2<T>*>(vertices->layout + vertexOffset1)->position;
    const T* pos_2 = reinterpret_cast<VertexGpuSe2<T>*>(vertices->layout + vertexOffset2)->position;

    // printf("%d) pos1: %f, %f, %f, %f, %f, %f \n", id1, pos_1[0], pos_1[1], pos_1[2], pos_1[3], pos_1[4], pos_1[5]);
    // printf("%d) pos2: %f, %f, %f, %f, %f, %f \n", id2, pos_2[0], pos_2[1], pos_2[2], pos_2[3], pos_2[4], pos_2[5]);
    // printf("%d - %d) meas: %f, %f, %f, %f, %f, %f \n", id1, id2, meas[0], meas[1], meas[2], meas[3], meas[4], meas[5]);

    T invPos[9];
    inverse(pos_1, invPos);

    T relPos[9];
    mul<T, 3, 3, 3>(invPos, pos_2, relPos);

    T invMeas[9];
    inverse(meas, invMeas);
    
    T delta[9];
    mul<T, 3, 3, 3>(invMeas, relPos, delta);

    error[0] = delta[2];
    error[1] = delta[5];
    error[2] = MatToAngle(delta);
}

template<typename T>
__global__ void ProcessSe2s(const VertexStorageGpu<T>* vertices, EdgeGpuSe2<T>* edges, const unsigned count,
                        T* H, T* b, T* totalError, const unsigned dims)
{
    const T deltaRobust = 1.5f;
    const T deltaRobustSqr = deltaRobust * deltaRobust;

    const int BLOCK_SIZE_1 = 3;
    const int BLOCK_SIZE_2 = 3;
    const auto INF_DIM = 3;

    const int edgeId = threadIdx.x + blockDim.x * blockIdx.x;

    for (int c = edgeId; c < count; c += blockDim.x * gridDim.x)
    {
        T errRobust = 0.0;
        T err[3];
        ProcessEdgeSe2<T>(vertices, edges[c].id1, edges[c].id2, 
            edges[c].meas, err);

        const T* INFOrig = edges[c].inf; // 3x3
        T INF[9]; // 3x3

        for (int k = 0; k < 9; ++k) INF[k] = INFOrig[k];

        T InfE[3];
        mulVec<T, 3, 3>(INF, err, InfE);

        T chi2 = 0.0;
        for (int ii = 0; ii < 3; ++ii)
        {
            chi2 += InfE[ii] * err[ii];
        }

        T err_J = 0.0;
        Robustify(chi2, deltaRobust, deltaRobustSqr, errRobust, err_J);
        INF[0] *= err_J;
        INF[4] *= err_J;
        INF[8] *= err_J;

        atomicAdd(totalError, errRobust);

        const auto index1 = vertices->indexesInH[edges[c].id1];
        const auto index2 = vertices->indexesInH[edges[c].id2];

        for (int o1 = 0; o1 < BLOCK_SIZE_1; ++o1)
        {
            for (int o2 = 0; o2 < BLOCK_SIZE_2; ++o2)
            {
                atomicAdd(&H[(index1 + o1) * dims + (index1 + o2)], INF[o1 + o2 * INF_DIM]);
                atomicAdd(&H[(index1 + o1) * dims + (index2 + o2)], -INF[o1 + o2 * INF_DIM]);
                atomicAdd(&H[(index2 + o1) * dims + (index1 + o2)], -INF[o1 + o2 * INF_DIM]);
                atomicAdd(&H[(index2 + o1) * dims + (index2 + o2)], INF[o1 + o2 * INF_DIM]);
            }
        }
        const T A[9] = {-1, 0, 0, 0, -1, 0, 0, 0, -1};
        const T B[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

        T INF_W[3];
        mulVec<T, 3, 3>(INF, err, INF_W);

        T AT_INF_W[BLOCK_SIZE_1]; 
        mulVec<T, 3, 3>(A, INF_W, AT_INF_W, true);

        T BT_INF_W[BLOCK_SIZE_2]; 
        mulVec<T, 3, 3>(B, INF_W, BT_INF_W, true);

        for (int ii = 0; ii < BLOCK_SIZE_1; ++ii) 
        {
            atomicAdd(&b[index1 + ii], AT_INF_W[ii]);
        }
        for (int ii = 0; ii < BLOCK_SIZE_2; ++ii) 
        {
            atomicAdd(&b[index2 + ii], BT_INF_W[ii]);
        }
    
        atomicAdd(totalError, errRobust);
    }
}

template __global__ void ProcessSe2s(const VertexStorageGpu<float>* vertices, EdgeGpuSe2<float>* edges, const unsigned count,
                        float* H, float* b, float* totalError, const unsigned dims);
template __global__ void ProcessSe2s(const VertexStorageGpu<double>* vertices, EdgeGpuSe2<double>* edges, const unsigned count,
                        double* H, double* b, double* totalError, const unsigned dims);