#pragma once

#include "graph/vertex/VertexType.h"

struct BaseVertexGpu
{
    VertexType type;
};

template<typename T, int SIZE>
struct BaseVertexGpuSized : public BaseVertexGpu
{
    T position[SIZE];
};

template<typename T>
struct VertexGpuSe2 : public BaseVertexGpuSized<T, 9>
{
};

template<typename T>
struct VertexGpuPoint2 : public BaseVertexGpuSized<T, 2>
{
};