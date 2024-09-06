#pragma once

#include "ConnectionHandlerGraph.h"
#include "GraphManager.h"

template<typename T>
void HandleConnection(std::shared_ptr<ConnectionHandlerGraph> handler, int iters, 
                      const OptimizerType optimizerType, const SolverType solverType) 
{
    std::cout << "\n------ New Connection ------\n";

    auto weakHandler = std::weak_ptr<ConnectionHandlerGraph>(handler);

    handler->SetGraphDataReceiveCallback([weakHandler, solverType, optimizerType, iters](std::vector<char>& bytes) {
            
            BlockTimer timer {"Total"};
            
            auto solver = CreateSolver<T>(solverType);
            auto optimizer = CreateOptimizer<T>(optimizerType, iters, std::move(solver));
            auto graph = CreateGraph<T>(optimizerType);
            DeserializeGraphStart<T>(graph.get(), bytes.data());

            optimizer->Optimize(graph.get());

            const auto data = SerializeGraphStart<T>(graph.get());

            if (auto handler = weakHandler.lock())
            {
                BlockTimer timer {"Sending"};
                handler->SendSync(data, [handler] {
                    handler->WaitForGraphSize();
                });
            }
        });

    // increases handler.use_count() 
    handler->WaitForGraphSize();
}

template<typename T, typename HandlerT>
void WaitForConnections(boost::asio::io_context& context, 
                        std::unique_ptr<ConnectionManagerServer<HandlerT>>& server,
                        int iters, const OptimizerType optimizerType, const SolverType solverType) 
{
    auto handler = std::make_shared<HandlerT>(context);

    server->SetOnNewConnectionCallback([iters, optimizerType, solverType](std::shared_ptr<HandlerT> new_handler) {
        HandleConnection<T>(new_handler, iters, optimizerType, solverType);
    });

    server->StartAccept(handler);
}