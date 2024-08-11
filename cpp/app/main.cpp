#include "core/conn/ConnectionManagerServer.h"
#include "core/matrix/MatrixHandler.h"
#include "GraphHandler.h"
#include "core/TypeDef.h"
#include "graph/Graph.h"
#include "graph/GraphOptimizer.h"
#include "graph/GraphDeserializer.h"
#include "graph/GraphSerializer.h"

#include <iostream>

Matrix2dShared DoSomeOp(const int cmd, Matrix2dShared& mat1, Matrix2dShared& mat2)
{
    if (cmd == 1)
        return mat1;

    return mat2;
}

int main(int argc, char *argv[])
{
    try
    {
        const std::string host = argc < 2 ? "127.0.0.1" : argv[1];
        const std::string port = argc < 3 ? "8888" : argv[2];
        const int optIterations = argc < 4 ? 10 : std::stoi(argv[3]);

        std::cout << "optIterations: " << optIterations << std::endl;

        boost::asio::io_context context;
        auto server = std::make_unique<ConnectionManagerServer>(context);
        server->Initialize(host, port);

        int command = -1;
        std::unique_ptr<Graph<float>> graph;

        auto handler = std::make_shared<GraphHandler>(context);
        handler->SetOnConnectCallback([handler]{
                handler->WaitForGraphSize();
            });
        handler->SetGraphDataReceiveCallback([handler, &command, &graph, &optIterations]
            (std::vector<char>& bytes) {
            graph = DeserializeGraph<float>(bytes.data());
            
            Optimize(graph.get(), optIterations);

            std::cout << "Serialize...\n";
            const auto data = SerializeGraph(graph.get());

            handler->SendSync(data, []{
                std::cout << "Sent result...\n";
            });
        });
        server->StartAccept(handler);

        context.run();
    }
    catch(std::exception& e)
    {
        std::cerr << "ConnectionManager error: " << e.what() << std::endl;
    }
  return 0;
}