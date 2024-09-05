#include "cuda/optimizer/kernels/KernelHelper.cuh"
#include "cuda/graph/VertexStorageGpu.h"
#include "cuda/graph/BaseVertexGpu.h"
#include "cuda/graph/BaseEdgeGpu.h"

template<typename T>
__device__ void ProcessEdgeSe2Point2(const VertexStorageGpu<T>* vertices, const unsigned id1, const unsigned id2,
                            const T* meas, T* error, T* A, T* B)
{
    const unsigned vertexOffset1 = vertices->offsets[id1];
    const unsigned vertexOffset2 = vertices->offsets[id2];
    const T* pos = reinterpret_cast<VertexGpuSe2<T>*>(vertices->layout + vertexOffset1)->position;
    const T* lm = reinterpret_cast<VertexGpuPoint2<T>*>(vertices->layout + vertexOffset2)->position;
    const T lmPos[3] = {lm[0], lm[1], 1};
    const T lm_local[2] = { meas[0] * cos(meas[1]), meas[0] * sin(meas[1]) };

    T invPos[9];
    inverse(pos, invPos);

    T pp[3];
    mulVec<T, 3, 3>(invPos, lmPos, pp);

    error[0] = pp[0] - lm_local[0];
    error[1] = pp[1] - lm_local[1];

    const auto x1 = pos[2 + 0 * 3];
    const auto y1 = pos[2 + 1 * 3];
    const auto th1 = MatToAngle(pos);
    const auto cosA = cos(th1);
    const auto sinA = sin(th1);

    A[0 * 3 +  0] = -cosA;
    A[0 * 3 +  1] = -sinA;
    A[0 * 3 +  2] = cosA * lm[1] - sinA * lm[0] - cosA * y1 + sinA * x1;
    A[1 * 3 +  0] = sinA;
    A[1 * 3 +  1] = -cosA;
    A[1 * 3 +  2] = -sinA * lm[1] - cosA * lm[0] + sinA * y1 + cosA * x1;

    B[0] = cosA;
    B[1] = sinA;
    B[2] = -sinA;
    B[3] = cosA;
}

template<typename T>
__global__ void ProcessSe2Point2s(const VertexStorageGpu<T>* vertices, EdgeGpuSe2Point2<T>* edges, const unsigned count,
                        T* H, T* b, T* totalError, const unsigned dims)
{
    constexpr T deltaRobust = 1.5f;
    constexpr T deltaRobustSqr = deltaRobust * deltaRobust;
    const int BLOCK_SIZE_1 = 3;
    const int BLOCK_SIZE_2 = 2;

    const int edgeId = threadIdx.x + blockDim.x * blockIdx.x;

    for (int c = edgeId; c < count; c += blockDim.x * gridDim.x)
    {
        const auto index1 = vertices->indexesInH[edges[c].id1];
        const auto index2 = vertices->indexesInH[edges[c].id2];

        T err[2];
        T A[6];
        T B[4];
        ProcessEdgeSe2Point2<T>(vertices, edges[c].id1, edges[c].id2, 
            edges[c].meas, err, A, B);

        const T* INFOrig = edges[c].inf; // 2x2
        T INF[4]; // 2x2

        for (int k = 0; k < 4; ++k) INF[k] = INFOrig[k];

        T InfE[2];
        mulVec<T, 2, 2>(INF, err, InfE);

        T chi2 = 0.0;
        for (int ii = 0; ii < 2; ++ii)
        {
            chi2 += InfE[ii] * err[ii];
        }

        T err_J = 0.0;
        T errRobust = 0.0;
        Robustify(chi2, deltaRobust, deltaRobustSqr, errRobust, err_J);
        INF[0] *= err_J;
        INF[3] *= err_J;

        T INF_W[2];
        mulVec<T, 2, 2>(INF, err, INF_W);

        // A.T * INF
        T AT_INF[BLOCK_SIZE_1 * BLOCK_SIZE_2]; 
        mul<T, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_2>(A, INF, AT_INF, true);

        // A.T * INF * A
        T ATINFA[BLOCK_SIZE_1 * BLOCK_SIZE_1]; 
        mul<T, BLOCK_SIZE_1, BLOCK_SIZE_1, BLOCK_SIZE_2>(AT_INF, A, ATINFA);

        // B.T * INF
        T BT_INF[BLOCK_SIZE_2 * BLOCK_SIZE_2]; 
        mul<T, BLOCK_SIZE_2, BLOCK_SIZE_2, BLOCK_SIZE_2>(B, INF, BT_INF, true);

        // B.T * INF * B
        T BT_INFB[BLOCK_SIZE_2 * BLOCK_SIZE_2]; 
        mul<T, BLOCK_SIZE_2, BLOCK_SIZE_2, BLOCK_SIZE_2>(BT_INF, B, BT_INFB);

        // A.T * INF * B
        T ATINFB[BLOCK_SIZE_1 * BLOCK_SIZE_2]; 
        mul<T, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_2>(AT_INF, B, ATINFB);

        // B.T * INF * A
        T BT_INFA[BLOCK_SIZE_2 * BLOCK_SIZE_1]; 
        transpose<T, BLOCK_SIZE_2, BLOCK_SIZE_1>(ATINFB, BT_INFA);

        for (int i = 0; i < BLOCK_SIZE_1; ++i) 
        {
            for (int j = 0; j < BLOCK_SIZE_1; ++j) 
            {
                atomicAdd(&H[(index1 + i) * dims + (index1 + j)], ATINFA[i * BLOCK_SIZE_1 + j]);
            }
            for (int j = 0; j < BLOCK_SIZE_2; ++j) 
            {
                atomicAdd(&H[(index1 + i) * dims + (index2 + j)], ATINFB[i * BLOCK_SIZE_2 + j]);
            }
        }
        for (int i = 0; i < BLOCK_SIZE_2; ++i) 
        {
            for (int j = 0; j < BLOCK_SIZE_1; ++j) 
            {
                atomicAdd(&H[(index2 + i) * dims + (index1 + j)], BT_INFA[i * BLOCK_SIZE_1 + j]);
            }
            for (int j = 0; j < BLOCK_SIZE_2; ++j) 
            {
                atomicAdd(&H[(index2 + i) * dims + (index2 + j)], BT_INFB[i * BLOCK_SIZE_2 + j]);
            }
        }
        
        T AT_INF_W[BLOCK_SIZE_1]; 
        mulVec<T, 2, 3>(A, INF_W, AT_INF_W, true);

        for (int ii = 0; ii < BLOCK_SIZE_1; ++ii) 
        {
            atomicAdd(&b[index1 + ii], AT_INF_W[ii]);
        }

        T BT_INF_W[BLOCK_SIZE_2]; 
        mulVec<T, 2, 2>(B, INF_W, BT_INF_W, true);
        
        for (int ii = 0; ii < BLOCK_SIZE_2; ++ii) 
        {
            atomicAdd(&b[index2 + ii], BT_INF_W[ii]);
        }

        atomicAdd(totalError, errRobust);
    }
}

template __global__ void ProcessSe2Point2s(const VertexStorageGpu<float>* vertices, EdgeGpuSe2Point2<float>* edges, const unsigned count,
                        float* H, float* b, float* totalError, const unsigned dims);
template __global__ void ProcessSe2Point2s(const VertexStorageGpu<double>* vertices, EdgeGpuSe2Point2<double>* edges, const unsigned count,
                        double* H, double* b, double* totalError, const unsigned dims);