#pragma once

#include "conn/ConnectionHandlerBase.h"

#include <boost/asio/io_context.hpp>

#include <memory>

template<typename HandlerT>
class ConnectionManagerServer
{
public:
    using NewConnectionCallback = std::function<void(std::shared_ptr<HandlerT>)>;

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
    void SetOnNewConnectionCallback(NewConnectionCallback callback) 
    {
        newConnectionCallback = std::move(callback);
    }

    void StartAccept(std::shared_ptr<HandlerT> handler) 
    {
        acceptor.async_accept(handler->GetSocket(),
            [this, handler](const boost::system::error_code& error) mutable {
                if (!error) 
                {
                    if (newConnectionCallback) 
                    {
                        newConnectionCallback(std::move(handler));
                    }
                }

                StartAccept(std::make_shared<HandlerT>(context));
            }
        );
    }

private:
    boost::asio::io_context& context;
    boost::asio::ip::tcp::acceptor acceptor;

    NewConnectionCallback newConnectionCallback;
};