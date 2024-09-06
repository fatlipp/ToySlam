#pragma once

#include "conn/ConnectionHandlerBase.h"

class ConnectionHandlerGraph : public ConnectionHandlerBase
{
public:
    ConnectionHandlerGraph(boost::asio::ip::tcp::socket socket)
        : ConnectionHandlerBase(std::move(socket))
        {}
    ConnectionHandlerGraph(boost::asio::io_context& context)
        : ConnectionHandlerBase(boost::asio::ip::tcp::socket(context))
        {}

public:
    void WaitForGraphSize()
    {
        bufferData.consume(bufferData.size());
        ReadAsync(bufferData, sizeof(int), 
            std::bind(&ConnectionHandlerGraph::ReceiveGraphSizeHandler, 
                shared_from_base<ConnectionHandlerGraph>()));
    }
    void WaitForGraph()
    {
        bufferData.consume(bufferData.size());
        ReadAsync(bufferData, graphDataSize, 
            std::bind(&ConnectionHandlerGraph::ReceiveGraphHandler, 
                shared_from_base<ConnectionHandlerGraph>()));
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

        WaitForGraph();
    }

    void ReceiveGraphHandler()
    {
        std::vector<char> bytes(graphDataSize, 0);
        std::istream inpStream(&bufferData);
        inpStream.read(bytes.data(), graphDataSize * sizeof(char));

        graphDataReceiveCallback(bytes);
    }

private:
    boost::asio::streambuf bufferData;

    int graphDataSize;

    std::function<void(std::vector<char>&)> graphDataReceiveCallback;
};