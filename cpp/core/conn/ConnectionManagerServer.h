#pragma once

#include "core/conn/ConnectionHandlerBase.h"

#include <boost/asio.hpp>
#include <memory>

class ConnectionManagerServer
{
public:
    ConnectionManagerServer(boost::asio::io_context& context)
        : context(context)
        , acceptor(context)
    {
    }

    void Initialize(const std::string& host, const std::string& port)
    {
        boost::asio::ip::tcp::resolver resolver(context);
        boost::asio::ip::tcp::resolver::results_type endpoints = resolver.resolve(host, port);

        std::cout << "Endpoints:\n";
        for (auto& e : endpoints)
        {
            std::cout << " - " << e.endpoint().address() << ":" << e.endpoint().port() << std::endl;
        }

        boost::asio::ip::tcp::endpoint endpoint = *endpoints.begin();
        acceptor.open(endpoint.protocol());
        acceptor.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        acceptor.bind(endpoint);
        acceptor.listen();
    }

public:
    void StartAccept(std::shared_ptr<ConnectionHandlerBase> handler)
    {
        auto& socket = handler->GetSocket();

        acceptor.async_accept(socket,
            [handler, &socket](const boost::system::error_code& ec) {
                if (ec) 
                {
                    std::cout << "A new connection error: " << ec.message() << std::endl;
                    return;
                }

                std::cout << "A new connection: " 
                    << socket.remote_endpoint().address() 
                    << ":" << socket.remote_endpoint().port() << '\n';

                handler->Ready();
            });
    }

private:
    boost::asio::io_context& context;
    boost::asio::ip::tcp::acceptor acceptor;
};