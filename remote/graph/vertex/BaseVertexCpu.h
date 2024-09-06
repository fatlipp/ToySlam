#pragma once

#include "graph/Types.h"
#include "graph/vertex/VertexType.h"

template<typename T>
class BaseVertexCpu
{
public:
    BaseVertexCpu(const VertexType type, const DynamicMatrix<T>& position)
        : type(type)
        , position{position}
    {
    }

    DynamicMatrix<T>& GetPosition()
    {
        return position;
    }

    const DynamicMatrix<T>& GetPosition() const
    {
        return position;
    }

    virtual void Update(const DynamicMatrix<T>& delta) = 0;

public:
    VertexType type;

protected:
    DynamicMatrix<T> position;
};
