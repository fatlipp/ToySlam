#pragma once

#include "graph/Types.h"

#include <Eigen/Eigen>

template<typename T>
class Graph;

template<typename T>
class BaseEdge
{
public:
    BaseEdge(const unsigned id1, const unsigned id2, 
        const DynamicMatrix<T>& meas, const DynamicMatrix<T>& inf)
        : id1{id1}
        , id2{id2}
        , meas{meas}
        , inf{inf}
    {
    }

    virtual std::tuple<DynamicVector<T>, DynamicMatrix<T>, DynamicMatrix<T>> 
                       CalcErrorAndJ(const Graph<T>* graph) = 0;


public:
    unsigned GetId1() const
    {
        return id1;
    }
    unsigned GetId2() const
    {
        return id2;
    }

    virtual std::string GetType() const = 0;

public:
    const unsigned id1; 
    const unsigned id2; 
    const DynamicMatrix<T> meas; 
    const DynamicMatrix<T> inf;

};
