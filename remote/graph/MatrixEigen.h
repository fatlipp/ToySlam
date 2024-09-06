#pragma once

#include "graph/IMatrix.h"
#include "graph/Types.h"

template<typename T>
class MatrixEigen : public IMatrix<T>
{
public:
    MatrixEigen(const unsigned rows, const unsigned cols)
        : IMatrix<T>(rows, cols)
    {
        data = Eigen::MatrixX<T>::Zero(rows, cols);
    }

    MatrixEigen(const unsigned rows, const unsigned cols, T* dataInp)
        : IMatrix<T>(rows, cols)
    {
        data = Eigen::MatrixX<T>::Zero(rows, cols);
        // data.data() = dataInp;
        Eigen::Map<Eigen::MatrixX<T>>(data.data(), rows, cols) = Eigen::Map<Eigen::MatrixX<T>>(dataInp, rows, cols);
    }

    void setZero() override
    {
        data.setZero();
    }

    const T* getData() const override
    {
        return data.data();
    }

    T* getData() override
    {
        return data.data();
    }

    const DynamicMatrix<T>& getEigen()
    {
        return data;
    }

    void AddBlock(const unsigned id1, const unsigned id2, 
        const unsigned dim1, const unsigned dim2, const DynamicMatrix<T>& val)
    {
        data.block(id1, id2, dim1, dim2) += val;
    }

private:
    DynamicMatrix<T> data;

};

template<typename T>
class VectorEigen : public IMatrix<T>
{
public:
    VectorEigen(const unsigned rows)
        : IMatrix<T>(rows, 1)
    {
        data = Eigen::VectorX<T>::Zero(rows);
    }

    VectorEigen(const unsigned rows, T* dataInp)
        : IMatrix<T>(rows, 1)
    {
        data = Eigen::VectorX<T>::Zero(rows);
        // data.data() = dataInp;
        Eigen::Map<Eigen::VectorX<T>>(data.data(), rows) = Eigen::Map<Eigen::VectorX<T>>(dataInp, rows);
    }

    void setZero() override
    {
        data.setZero();
    }

    const T* getData() const override
    {
        return data.data();
    }

    T* getData() override
    {
        return data.data();
    }

    DynamicVector<T>& getEigen()
    {
        return data;
    }

    void AddSegment(const unsigned id1,  const unsigned dim1, const DynamicVector<T>& val)
    {
        data.segment(id1, dim1) -= val;
    }

private:
    DynamicVector<T> data;

};