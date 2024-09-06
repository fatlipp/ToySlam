#pragma once

#include "graph/edge/EdgeType.h"

class BaseEdgeGpu
{
public:
    EdgeType type;
    unsigned id1;
    unsigned id2;
};

template<typename T, int SIZE_MEAS, int SIZE_INF>
struct BaseEdgeGpuSized : public BaseEdgeGpu
{
    T meas[SIZE_MEAS];
    T inf[SIZE_INF]; 
};

template<typename T>
struct EdgeGpuSe2 : public BaseEdgeGpuSized<T, 3*3, 3*3>
{
};

template<typename T>
struct EdgeGpuSe2Point2 : public BaseEdgeGpuSized<T, 2*1, 2*2>
{
};