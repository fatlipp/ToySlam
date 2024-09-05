#pragma once

#include <memory>

class IGraph;

template<typename T>
class ISolver;

template<typename T>
class IOptimizer 
{
public:
    IOptimizer(const unsigned iterations, std::unique_ptr<ISolver<T>> solver)
        : iterations{iterations}
        , solver{std::move(solver)}
    {
    }

    virtual ~IOptimizer() = default;
    virtual void Optimize(IGraph* graph) = 0;

protected:
    unsigned iterations;
    std::unique_ptr<ISolver<T>> solver;
};