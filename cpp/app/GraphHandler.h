#pragma once

#include "core/conn/ConnectionHandlerBase.h"
#include "core/TypeDef.h"

#include <iostream>
#include <thread>

#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>


class GraphHandler : public ConnectionHandlerBase
{
public:
    GraphHandler(boost::asio::ip::tcp::socket socket)
        : ConnectionHandlerBase(std::move(socket))
        {}
    GraphHandler(boost::asio::io_context& context)
        : ConnectionHandlerBase(boost::asio::ip::tcp::socket(context))
        {}

public:
    void WaitForGraphSize()
    {
        bufferData.consume(bufferData.size());
        ReadAsync(bufferData, sizeof(int), 
            std::bind(&GraphHandler::ReceiveGraphSizeHandler, 
                shared_from_base<GraphHandler>()));
    }
    void WaitForGraph()
    {
        bufferData.consume(bufferData.size());
        ReadAsync(bufferData, graphDataSize, 
            std::bind(&GraphHandler::ReceiveGraphHandler, 
                shared_from_base<GraphHandler>()));
    }

    void SetGraphDataReceiveCallback(const std::function<void(std::vector<char>&)>& cb)
    {
        graphDataReceiveCallback = cb;
    }

private:
    void ReceiveGraphSizeHandler()
    {
        std::istream inpStream(&bufferData);
        inpStream.read(reinterpret_cast<char*>(&graphDataSize), sizeof(int));

        std::cout << "graphDataSize: " 
            << graphDataSize << std::endl;

        WaitForGraph();
    }

    void ReceiveGraphHandler()
    {
        std::cout << "" << "ReceiveGraphHandler" << std::endl;
        std::vector<char> bytes(graphDataSize, 0);
        std::istream inpStream(&bufferData);
        std::cout << "" << "ReceiveGraphHandler 2" << std::endl;
        inpStream.read(bytes.data(), graphDataSize * sizeof(char));

        graphDataReceiveCallback(bytes);
    }

private:
    boost::asio::streambuf bufferData;

    int graphDataSize;

    std::function<void(std::vector<char>&)> graphDataReceiveCallback;
};