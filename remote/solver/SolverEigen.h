#pragma once

#include "solver/ISolver.h"
#include "graph/Types.h"
#include "graph/MatrixEigen.h"

template<typename T>
class SolverEigen : public ISolver<T>
{
public:
    void Solve(IMatrix<T>* H, IMatrix<T>* b, unsigned dims) override
    {
        static_assert(std::is_base_of_v<IMatrix<T>, MatrixEigen<T>>, 
                "MatrixEigen<T> must derive from IMatrix<T>");
        static_assert(std::is_base_of_v<IMatrix<T>, VectorEigen<T>>, 
                "VectorEigen<T> must derive from IMatrix<T>");
        MatrixEigen<T>* EigenH = dynamic_cast<MatrixEigen<T>*>(H);
        VectorEigen<T>* EigenB = dynamic_cast<VectorEigen<T>*>(b);

        EigenB->getEigen() = EigenH->getEigen().colPivHouseholderQr().solve(EigenB->getEigen());
    }
};