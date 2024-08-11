#pragma once

#include "core/matrix/MatrixHandler.h"
#include "core/TypeDef.h"

#include <boost/asio.hpp>
#include <iostream>
#include <memory>

class ConnectionManagerClient : public MatrixHandler
{
public:
    ConnectionManagerClient(boost::asio::io_context& context)
        : MatrixHandler(context)
        , resolver(context)
    { 
    }

    void Initialize(const std::string& host, const std::string& port)
    {
        resolver.async_resolve(host, port,
            [self = shared_from_base<ConnectionManagerClient>()]
            (const boost::system::error_code& err, 
                    boost::asio::ip::tcp::resolver::results_type endpoints) {
                if (err) 
                {
                    std::cerr << "Write failed: " << err.message() << "\n";
                    self->Close();
                    return;
                }

                self->Connect(endpoints);
            });
    }

private:
    void Connect(const boost::asio::ip::tcp::resolver::results_type& endpoints) 
    {
        boost::asio::async_connect(GetSocket(), endpoints,
            [self = shared_from_this()](const boost::system::error_code& err, 
                   const boost::asio::ip::tcp::endpoint& endpoint) {
                if (err) 
                {
                    std::cerr << "Connect failed: " << err.message() << "\n";
                    self->Close();
                    return;
                }
                
                std::cout << "Connected to: " << endpoint.address() << ":" << endpoint.port() << std::endl;

                self->Ready();
            });
    }

private:
    boost::asio::ip::tcp::resolver resolver;
};