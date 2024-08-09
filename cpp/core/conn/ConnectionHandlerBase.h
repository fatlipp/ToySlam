#pragma once

#include "core/conn/SharedFromBase.h"

#include <iostream>

#include <boost/asio.hpp>
#include <boost/function.hpp>

class ConnectionHandlerBase : public enable_shared_from_base<ConnectionHandlerBase>
{
public:
    ConnectionHandlerBase(boost::asio::ip::tcp::socket socket)
        : socket {std::move(socket)}
        {}

    virtual ~ConnectionHandlerBase() = default;

public:
    virtual void Close() 
    {
        auto self = shared_from_this();
        boost::asio::post(GetSocket().get_executor(), [self]() { 
                self->GetSocket().close();
                std::cout << "Disconnected()" << std::endl;
            });
    }

    void Ready()
    {
        if (onConnectCallback)
        {
            onConnectCallback();
        }
    }

    void ReadAsync(boost::asio::streambuf& buffer, const int size, 
        const boost::function<void()>& callback = nullptr)
    {
        auto self = shared_from_base<ConnectionHandlerBase>();

        boost::asio::async_read(GetSocket(), buffer, boost::asio::transfer_exactly(size),
            [self, callback](const auto& err, const auto& size) {
                if (err) 
                {
                    std::cerr << "ReadAsync() error: " << err.message() << std::endl;
                    self->Close();
                    return;
                }

                if (callback)
                {
                    callback();
                }
            });
    }

    void SendAsync(boost::asio::streambuf& buffer, const int size,
        const std::function<void()>& callback = nullptr)
    {
        auto self = shared_from_base<ConnectionHandlerBase>();

        boost::asio::async_write(GetSocket(), buffer,
            [self, callback, &buffer, size](const auto& err, const auto& sentBytes) {
                if (err) 
                {
                    std::cerr << "SendAsync() error: " << err.message() << std::endl;
                    self->Close();
                    return;
                }
        
                if (sentBytes != size)
                {
                    std::cerr << "SendAsync() error: " << sentBytes << " of " << size << std::endl;
                    self->Close();
                    return;
                }

                if (callback)
                {
                    callback();
                }
            });
    }

    void SendSync(boost::asio::streambuf& buffer, const int size,
        const std::function<void()>& callback = nullptr)
    {
        boost::system::error_code error;
        const auto sentBytes = boost::asio::write(GetSocket(), buffer, error);

        if (error)
        {
            std::cerr << "SendSync() error: " << error.message() << std::endl;
            Close();
        }
        else if (sentBytes != size)
        {
            std::cerr << "SendSync() error: " << sentBytes << " of " << size << std::endl;
        }
        else if (callback)
        {
            callback();
        }
    }

    void SendSync(const std::vector<uint8_t>& data,
        const std::function<void()>& callback = nullptr)
    {    
        std::cout << "SendSync = " << data.size() << std::endl;
        boost::asio::streambuf streambuf;
        std::size_t bufferSize = data.size();
        streambuf.prepare(bufferSize);

        // Copy the char array into the streambuf
        boost::asio::buffer_copy(streambuf.prepare(bufferSize), boost::asio::buffer(data.data(), bufferSize));
        
        streambuf.commit(bufferSize);

        std::cout << "bufferSize = " << bufferSize << std::endl;

        SendSync(streambuf, bufferSize, callback);
    }
    
public:
    boost::asio::ip::tcp::socket& GetSocket()
    {
        return socket;
    }

    void SetOnConnectCallback(const std::function<void()>& onConnectCallback)
    {
        this->onConnectCallback = onConnectCallback;
    }

private:
    boost::asio::ip::tcp::socket socket;

    std::function<void()> onConnectCallback;
};