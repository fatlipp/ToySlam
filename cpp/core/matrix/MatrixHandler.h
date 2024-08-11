#pragma once

#include "core/conn/ConnectionHandlerBase.h"
#include "core/TypeDef.h"

#include <iostream>
#include <thread>

#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>


class MatrixHandler : public ConnectionHandlerBase
{
public:
    MatrixHandler(boost::asio::ip::tcp::socket socket)
        : ConnectionHandlerBase(std::move(socket))
        {}
    MatrixHandler(boost::asio::io_context& context)
        : ConnectionHandlerBase(boost::asio::ip::tcp::socket(context))
        {}

public:
    void WaitForMatrix()
    {
        bufferData.consume(bufferData.size());

        ReadAsync(bufferData, sizeof(int) * 2, 
            std::bind(&MatrixHandler::ReadShapeHandler, 
                shared_from_base<MatrixHandler>()));
    }

    void SetMatrixReceiveCallback(const std::function<void(Matrix2dShared)>& cb)
    {
        onMatrixReceived = cb;
    }

    void SendMatrix2d(Matrix2dShared matrix, const std::function<void()>& cb = nullptr)
    {
        if (matrix == nullptr)
        {
            if (cb)
            {
                cb();
            }

            return;
        }
        const int rows = matrix->size();
        const int cols = matrix->at(0).size();

        bufferData.consume(bufferData.size());
        std::ostream os(&bufferData);
        os.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        os.write(reinterpret_cast<const char*>(&cols), sizeof(int));

        auto self = shared_from_base<MatrixHandler>();
        SendAsync(bufferData, sizeof(int) * 2,
            [self, matrix, rows, cols, cb] {
                
                for (int i = 0; i < rows; ++i)
                {
                    boost::asio::streambuf buffer;
                    self->CopyBuffer(boost::asio::buffer(matrix->at(i)), buffer);
                    self->SendSync(buffer, cols * sizeof(float));
                }

                if (cb)
                {
                    cb();
                }
            });
    }

private:
    void ReadShapeHandler()
    {
        std::istream inpStream(&bufferData);
        inpStream.read(reinterpret_cast<char*>(&rows), sizeof(int));
        inpStream.read(reinterpret_cast<char*>(&cols), sizeof(int));

        std::cout << "|-ReadShapeHandler() Shape: " << rows << " x " << cols << std::endl;

        if (rows > 10000 || cols > 10000)
        {
            std::cout << "Matrix is too big" << std::endl;
            Close();
            return;
        }
        bufferData.consume(bufferData.size());

        ReadAsync(bufferData, cols * rows * sizeof(float),
            std::bind(&MatrixHandler::ReadDataHandler, shared_from_base<MatrixHandler>()));
    }

    void ReadDataHandler()
    {
        std::cout << "|--ReadDataHandler(): OK" << std::endl;

        if (rows == 1)
        {
            ReadVector();
        }
        else
        {
            ReadMatrix();
        }
    }

    void ReadVector()
    {
        auto matrix2d = std::make_shared<Matrix2d>();
        matrix2d->resize(1);
        matrix2d->at(0).resize(cols);

        const auto begin = boost::asio::buffers_begin(bufferData.data());
        const int length = cols * sizeof(float);

        std::memcpy(matrix2d->at(0).data(), &(*begin), length);

        if (onMatrixReceived)
        {
            onMatrixReceived(matrix2d);
        }
    }

    void ReadMatrix()
    {
        auto matrix2d = std::make_shared<Matrix2d>();
        matrix2d->resize(rows);

        const auto begin = boost::asio::buffers_begin(bufferData.data());
        const int length = cols * sizeof(float);

        for (int i = 0; i < rows; ++i)
        {   
            const int startOffset = i * length;

            matrix2d->at(i).resize(cols);
    
            std::memcpy(matrix2d->at(i).data(), &(*(begin + startOffset)), length);
        }

        if (onMatrixReceived)
        {
            onMatrixReceived(matrix2d);
        }
    }

    void CopyBuffer(const boost::asio::const_buffer& source, boost::asio::streambuf& target)
    {
        std::ostream os(&target);
        os.write(boost::asio::buffer_cast<const char*>(source), boost::asio::buffer_size(source));
    }

private:
    int rows;
    int cols;
    boost::asio::streambuf bufferData;

    std::function<void(Matrix2dShared)> onMatrixReceived;
};