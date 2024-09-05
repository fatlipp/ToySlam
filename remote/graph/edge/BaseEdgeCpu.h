#pragma once

#include "graph/edge/EdgeType.h"
#include "graph/Types.h"

template<typename T>
class GraphCpu;

template<typename T>
class BaseEdgeCpu
{
public:
    BaseEdgeCpu(const EdgeType type, const unsigned id1, const unsigned id2, 
        const DynamicMatrix<T>& meas, const DynamicMatrix<T>& inf)
        : type{type}
        , id1{id1}
        , id2{id2}
        , meas{meas}
        , inf{inf}
    {
    }

    virtual std::tuple<DynamicVector<T>, DynamicMatrix<T>, DynamicMatrix<T>> 
                       CalcErrorAndJ(const GraphCpu<T>* graph) = 0;


public:
    const DynamicMatrix<T>& GetMeas() const
    {
        return meas;
    }
    const DynamicMatrix<T>& GetInf() const
    {
        return inf;
    }

public:
    EdgeType type;
    unsigned id1;
    unsigned id2;

protected:
    const DynamicMatrix<T> meas; 
    const DynamicMatrix<T> inf;

};
