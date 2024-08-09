#pragma once

#include "graph/Graph.h"
#include "graph/Types.h"


template<typename T>
static void Optimize(Graph<T>* graph, const int iteration)
{
    if (graph == nullptr)
    {
        return;
    }

    auto robustify = [](const float& e,const float delta) -> std::pair<float, float>
    {
        const auto deltaSqr = std::pow(delta, 2);
        if (e <= deltaSqr)
        {
            return {e, 1};
        }

        const auto sqrte = std::sqrt(e);
        return {2 * sqrte * delta - deltaSqr, delta / sqrte};
    };

    auto updateLambda = [](const bool isErrIncrease, const float lambda) {
                const auto lambdaMin = 1e-6; 
                const auto lambdaMax = 1e1;
                const auto lambdaFactor = 1.1;

                if (isErrIncrease)
                {
                    return std::min(lambda * lambdaFactor, lambdaMax);
                }

                return std::max(lambda / lambdaFactor, lambdaMin);
            };

    std::unordered_map<unsigned, unsigned> vertex_ids_map;
    unsigned dimension = 0;
    for (const auto& [id, v] : graph->GetVertices())
    {
        vertex_ids_map[id] = dimension;
        dimension += v->GetDims();
    }

    DynamicMatrix<T> H = DynamicMatrix<T>::Zero(dimension, dimension);
    DynamicVector<T> b = DynamicVector<T>::Zero(dimension);

    std::cout << "H: " << H.rows() << "x" << H.cols() << "\n";
    std::cout << "b: " << b.rows() << "\n";


    auto lambdaVal = 1e-3;

    float prevErr = -1;
    int penalty = 0;

    for (int i = 0; i < iteration; ++i)
    {
        double err = 0;
        H.setZero();
        b.setZero();

        for (const auto& edge : graph->GetEdges())
        {
            const auto [e, A, B] = edge->CalcErrorAndJ(graph);
            auto chi_2 = e.dot(edge->inf * e);
            auto [er, err_J] = robustify(chi_2, 1.5f);
            auto INF = edge->inf * err_J;
            auto INF_W = INF * e;
            // std::cout << "Edge: " << edge->id1 << " -> " << edge->id2 
            //           << " e = " << er << std::endl;

            const auto index1 = vertex_ids_map[edge->id1];
            const auto index2 = vertex_ids_map[edge->id2];
            const int BLOCK_SIZE_1 = graph->GetVertex(edge->id1)->GetDims();
            const int BLOCK_SIZE_2 = graph->GetVertex(edge->id2)->GetDims();

            H.block(index1, index1, BLOCK_SIZE_1, BLOCK_SIZE_1) += (A.transpose() * INF * A);
            H.block(index2, index2, BLOCK_SIZE_2, BLOCK_SIZE_2) += (B.transpose() * INF * B);
            H.block(index1, index2, BLOCK_SIZE_1, BLOCK_SIZE_2) += (A.transpose() * INF * B);
            H.block(index2, index1, BLOCK_SIZE_2, BLOCK_SIZE_1) += (B.transpose() * INF * A);
            b.segment(index1, BLOCK_SIZE_1) += A.transpose() * INF_W;
            b.segment(index2, BLOCK_SIZE_2) += B.transpose() * INF_W;
            err += er;
        }

        for (const auto& id : graph->GetFixedVertices())
        {
            const auto index = vertex_ids_map[id];
            const int SIZE = graph->GetVertex(id)->GetDims();
            H.block(index, index, SIZE, SIZE) += Eigen::MatrixXf::Identity(SIZE, SIZE) * 1e6;
            b.segment(index, SIZE) = Eigen::MatrixXf::Zero(SIZE, SIZE);
        }

        std::cout << "Total err = " << err << std::endl;

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

        // lambdaVal = updateLambda(penalty > 0, lambdaVal);
        // H = H + lambdaVal * Eigen::MatrixXf::Identity(dimension, dimension);


        const Eigen::VectorXf delta = H.colPivHouseholderQr().solve(-b) * 0.2;

        for (auto& [id, v] : graph->GetVertices())
        {
            const auto index = vertex_ids_map[id];
            const int SIZE = v->GetDims();

            v->Update(delta.segment(index, SIZE));
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
}