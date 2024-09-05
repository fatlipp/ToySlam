#pragma once

#include "optimizer/IOptimizer.h"
#include "solver/ISolver.h"

template<typename T>
class OptimizerGpu : public IOptimizer<T>
{
public:
    OptimizerGpu(const unsigned iterations, std::unique_ptr<ISolver<T>> solver)
        : IOptimizer<T>(iterations, std::move(solver))
    {
    }

    void Optimize(IGraph* graph) override;
};
