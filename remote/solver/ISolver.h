#pragma once

#include "graph/IMatrix.h"

template<typename T>
class ISolver 
{
public:
    virtual ~ISolver() = default;
    virtual void Solve(IMatrix<T>* H, IMatrix<T>* b, unsigned dims) = 0;
};
