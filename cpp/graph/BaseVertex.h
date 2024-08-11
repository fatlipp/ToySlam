#pragma once

#include "graph/Types.h"

#include <Eigen/Eigen>

template<typename T>
class BaseVertex
{
public:
    BaseVertex(const DynamicMatrix<T>& position)
        : position{position}
    {
    }

    virtual unsigned GetDims() const = 0;
    virtual std::string GetType() const = 0;

    virtual void Update(const DynamicMatrix<T>& delta) = 0;

public:
    DynamicMatrix<T> position;

};