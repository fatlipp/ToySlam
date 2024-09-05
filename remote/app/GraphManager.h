#pragma once

#include "graph/GraphCpu.h"
#include "solver/SolverEigen.h"
#include "optimizer/IOptimizer.h"
#include "optimizer/OptimizerCpu.h"
#include "serialization/SerializeGraph.h"
#include "serialization/DeserializeGraph.h"
#include "serialization/DeserializeGraphFuncCpu.h"
#include "serialization/SerializeGraphFuncCpu.h"

#include <iostream>

#ifdef WITH_CUDA
#include "solver/SolverCpuMatWithCuda.h"

#include "cuda/graph/GraphGpu.h"
#include "cuda/optimizer/OptimizerGpu.h"
#include "cuda/serialization/DeserializeGraphFuncGpu.h"
#include "cuda/serialization/SerializeGraphFuncGpu.h"
#endif

enum class SolverType
{
    EIGEN,
    CUDA
};

enum class OptimizerType
{
    EIGEN,
    CUDA
};

template<typename T>
void DeserializeGraphStart(IGraph* graph, char* ptr)
{
    if (graph == nullptr)
    {
        std::cerr << "NULL graph\n";
        return;
    }

    if (dynamic_cast<GraphCpu<T>*>(graph))
    {
        DeserializeGraph<T, GraphCpu<T>, DeserializeGraphFuncCpu<T>>(dynamic_cast<GraphCpu<T>*>(graph), ptr);
    }
#ifdef WITH_CUDA
    else if (dynamic_cast<GraphGpu<T>*>(graph))
    {
        DeserializeGraph<T, GraphGpu<T>, DeserializeGraphFuncGpu<T>>(dynamic_cast<GraphGpu<T>*>(graph), ptr);
    }
#endif
}

template<typename T>
std::vector<uint8_t> SerializeGraphStart(IGraph* graph)
{
    if (dynamic_cast<GraphCpu<T>*>(graph))
    {
        return SerializeGraph<T, GraphCpu<T>, SerializeGraphFuncCpu<T>>(dynamic_cast<GraphCpu<T>*>(graph));
    }
#ifdef WITH_CUDA
    else if (dynamic_cast<GraphGpu<T>*>(graph))
    {
        return SerializeGraph<T, GraphGpu<T>, SerializeGraphFuncGpu<T>>(dynamic_cast<GraphGpu<T>*>(graph));
    }
#endif

    return {};
}

template<typename T>
std::unique_ptr<IGraph> CreateGraph(const OptimizerType optimizerType)
{
    BlockTimer timer {"CreateGraph", 1};
#ifdef WITH_CUDA
    if (optimizerType != OptimizerType::EIGEN)
    {
        return std::make_unique<GraphGpu<T>>();
    }
#endif

    return std::make_unique<GraphCpu<T>>();
}

template<typename T>
std::unique_ptr<ISolver<T>> CreateSolver(const SolverType solverType) 
{
    BlockTimer timer {"CreateSolver", 1};
    if (solverType == SolverType::EIGEN) 
    {
        return std::make_unique<SolverEigen<T>>();
    }
#ifdef WITH_CUDA
    else if (solverType == SolverType::CUDA) 
    {
        return std::make_unique<SolverCpuMatWithCuda<T>>();
    }
#endif
    
    return nullptr;
}

template<typename T>
std::unique_ptr<IOptimizer<T>> CreateOptimizer(const OptimizerType optimizerType,
    const unsigned iters, std::unique_ptr<ISolver<T>> solver) 
{
    BlockTimer timer {"CreateOptimizer", 1};
    if (optimizerType == OptimizerType::EIGEN) 
    {
        return std::make_unique<OptimizerCpu<T>>(iters, std::move(solver));
    } 
#ifdef WITH_CUDA
    else if (optimizerType == OptimizerType::CUDA) 
    {
        return std::make_unique<OptimizerGpu<T>>(iters, std::move(solver));
    }
#endif
    
    return nullptr;
}
