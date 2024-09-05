#pragma once

#include "optimizer/IOptimizer.h"

#include "graph/GraphCpu.h"
#include "graph/edge/BaseEdgeCpu.h"
#include "graph/MatrixEigen.h"
#include "graph/Types.h"
#include "graph/Helper.h"
#include "optimizer/ThreadPool.h"
#include "tools/BlockTimer.h"

#include <iomanip>
#include <mutex>

template<typename T>
class OptimizerCpu : public IOptimizer<T>
{
public:
    OptimizerCpu(const unsigned iterations, std::unique_ptr<ISolver<T>> solver)
        : IOptimizer<T>(iterations, std::move(solver))
    {
    }

    void Optimize(IGraph* graphIn) override
    {
        if (graphIn == nullptr)
        {
            return;
        }

        BlockTimer timer{"OptimizeCPU"};

        auto graph = dynamic_cast<GraphCpu<T>*>(graphIn);

        auto robustify = [](const T& e, const T& delta) -> std::pair<T, T>
        {
            const auto deltaSqr = std::pow(delta, 2);
            if (e <= deltaSqr)
            {
                return {e, 1};
            }

            const auto sqrte = std::sqrt(e);
            return {2 * sqrte * delta - deltaSqr, delta / sqrte};
        };

        auto getDims = [](const BaseVertexCpu<T>* v) {

                switch (v->type)
                {
                case VertexType::Se2:
                    return 3;
                
                default:
                    break;
                }

                return 2;
            };

        std::unordered_map<unsigned, unsigned> vertex_ids_map;
        unsigned dimension = 0;
        for (const auto& [id, v] : graph->GetVertices())
        {
            vertex_ids_map[id] = dimension;
            dimension += getDims(v.get());
        }

        auto lambdaVal = 1e-3;
        T prevErr = -1;
        T err = 0;
        int penalty = 0;

         MatrixEigen<T> H{dimension, dimension};
         VectorEigen<T> b{dimension};

        // ThreadPool pool{8};

        for (int i = 0; i < this->iterations; ++i)
        {
            err = 0;
            H.setZero();
            b.setZero();

            std::mutex solverMutex;

            auto edgeProcessor = [&graph, &H, &b, &err, &robustify, &getDims,
                &solverMutex, &vertex_ids_map](BaseEdgeCpu<T>* edge) { 
                    const auto [e, A, B] = edge->CalcErrorAndJ(graph);
                    const auto chi_2 = e.dot(edge->GetInf() * e);
                    const auto [er, err_J] = robustify(chi_2, 1.5f);
                    auto INF = edge->GetInf() * err_J;
                    auto INF_W = INF * e;

                    const auto index1 = vertex_ids_map[edge->id1];
                    const auto index2 = vertex_ids_map[edge->id2];
                    const int BLOCK_SIZE_1 = getDims(graph->GetVertex(edge->id1));
                    const int BLOCK_SIZE_2 = getDims(graph->GetVertex(edge->id2));

                    auto blockAtA = A.transpose() * INF * A;
                    auto blockBtB = B.transpose() * INF * B;
                    auto block1 = A.transpose() * INF * B;
                    auto block2 = B.transpose() * INF * A;

                    auto pos_1 = graph->GetVertex(edge->id1)->GetPosition().data();
                    auto pos_2 = graph->GetVertex(edge->id2)->GetPosition().data();

                    {
                        std::lock_guard<std::mutex> lock(solverMutex);
                        H.AddBlock(index1, index1, BLOCK_SIZE_1, BLOCK_SIZE_1, blockAtA);
                        H.AddBlock(index2, index2, BLOCK_SIZE_2, BLOCK_SIZE_2, blockBtB);
                        H.AddBlock(index1, index2, BLOCK_SIZE_1, BLOCK_SIZE_2, block1);
                        H.AddBlock(index2, index1, BLOCK_SIZE_2, BLOCK_SIZE_1, block2);
                        b.AddSegment(index1, BLOCK_SIZE_1, A.transpose() * INF_W);
                        b.AddSegment(index2, BLOCK_SIZE_2, B.transpose() * INF_W);
                        err += er;
                    }
                };

            for (const auto& [type, edges] : graph->GetEdges())
            {
                for (auto& edge : edges)
                {
                // pool.AddTask([&edge, &edgeProcessor](){
                        edgeProcessor(edge.get());
                    // });    
                }
            }
            // pool.Wait();

            for (const auto& id : graph->GetFixedVertices())
            {
                const auto index = vertex_ids_map[id];
                const int SIZE = getDims(graph->GetVertex(id));
                H.AddBlock(index, index, SIZE, SIZE, Eigen::MatrixX<T>::Identity(SIZE, SIZE) * 1e6);
                b.AddSegment(index, SIZE, Eigen::VectorX<T>::Zero(SIZE));
            }

            if (prevErr > 0 && err > prevErr)
            {
                ++penalty;

                if (penalty > 2)
                {
                    std::cout << "Error is getting worse\n";
                    break;
                }
            }
            else
            {
                penalty = 0;
            }

            this->solver->Solve(&H, &b, dimension);

            const auto delta = b.getEigen();

            for (auto& [id, v] : graph->GetVertices())
            {
                const auto index = vertex_ids_map[id];
                const int SIZE = getDims(v.get());

                v->Update(delta.segment(index, SIZE) * 0.2);
            }

            if (std::abs(err - prevErr) < 0.001)
            {
                std::cout << "Plateau: NO MORE OPT\n";
                break;
            }

            if (delta.norm() < 0.001)
            {
                std::cout << "CONVERGED\n";
                break;
            }

            prevErr = err;
        }

        std::cout << "Summary() error = " << err << std::endl;
    }
};