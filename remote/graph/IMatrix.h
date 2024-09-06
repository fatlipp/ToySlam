#pragma once

template<typename T>
class IMatrix
{
public:
    IMatrix()
        : cols{0}
        , rows{0}
    {}

    IMatrix(const unsigned rows, const unsigned cols)
        : rows{rows}
        , cols{cols}
    {}

    virtual ~IMatrix() = default;

public:
    virtual void setZero() = 0;

    virtual unsigned getRows() { return rows; }
    virtual unsigned getCols() { return cols; }
    virtual const T* getData() const = 0;
    virtual T* getData() = 0;

protected:
    unsigned rows;
    unsigned cols;
};
