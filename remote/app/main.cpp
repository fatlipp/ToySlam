#include "conn/ConnectionManagerServer.h"
#include "ConnectionHandler.h"


int main(int argc, char *argv[])
{
#ifdef WITH_CUDA
    std::cout << "CUDA is supported\n";
#endif
    try
    {
        const std::string host = argc < 2 ? "127.0.0.1" : argv[1];
        const std::string port = argc < 3 ? "8888" : argv[2];
        const int iters = argc < 4 ? 10 : std::stoi(argv[3]);
        const std::string targetS = argc < 5 ? "cpu" : argv[4];
        const std::string solverS = argc < 6 ? "eigen" : argv[5];

        auto optimizerType = targetS == "cpu" ? OptimizerType::EIGEN : OptimizerType::CUDA;
        auto solverType = solverS == "eigen" ? SolverType::EIGEN : SolverType::CUDA;

#ifdef WITH_CUDA
        if (optimizerType == OptimizerType::CUDA)
        {
            solverType = SolverType::CUDA;
        }
#else
        optimizerType = OptimizerType::EIGEN;
        solverType = SolverType::EIGEN;
#endif

        std::cout << "iters: " << iters  << 
            ", optimizerType: " << static_cast<int>(optimizerType) << 
            ", solverType: " << static_cast<int>(solverType) << 
            std::endl;

        boost::asio::io_context context;
        auto server = std::make_unique<ConnectionManagerServer<ConnectionHandlerGraph>>(context);
        server->Initialize(host, port);

        WaitForConnections<float, ConnectionHandlerGraph>(context, server, iters, optimizerType, solverType);

        context.run();
    }
    catch(std::exception& e)
    {
        std::cerr << "ConnectionManager error: " << e.what() << std::endl;
    }
  return 0;
}