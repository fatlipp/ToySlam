#pragma once

#include "graph/Graph.h"
#include "graph/EdgeSe2.h"
#include "graph/EdgeSe2Point2d.h"
#include "graph/Vertex2d.h"
#include "graph/VertexSe2.h"
#include "graph/Helper.h"

#include <iostream>

#include <Eigen/Eigen>

#include <vector>
#include <string>

template<typename T>
std::unique_ptr<Graph<T>> DeserializeGraph(char* ptr) 
{
    std::unique_ptr<Graph<T>> graph = std::make_unique<Graph<T>>();

    using DATA_TYPE = float;
    const int STEP_4 = 4;
    const int STEP_8 = sizeof(DATA_TYPE);
    const auto verticesCount = *reinterpret_cast<unsigned*>(ptr);
    std::cout << "Vertices count: " << verticesCount << std::endl;
    ptr += STEP_4;

    int globalLmId = 0;

    for (unsigned i = 0; i < verticesCount; ++i)
    {
        const unsigned id = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;
        const unsigned typeStrSize = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;
        std::string typeStr {ptr, ptr + typeStrSize};
        ptr += typeStrSize;
        
        // std::cout << i << ") typeStr: " << typeStr << "";
        if (typeStr == "pose2d")
        {
            const DATA_TYPE x = *reinterpret_cast<DATA_TYPE*>(ptr);
            ptr += STEP_8;
            const DATA_TYPE y = *reinterpret_cast<DATA_TYPE*>(ptr);
            ptr += STEP_8;
            const DATA_TYPE theta = *reinterpret_cast<DATA_TYPE*>(ptr);
            ptr += STEP_8;

            const auto posMat = CreateTransform2d(x, y, theta);

            graph->AddVertex(id, std::make_unique<VertexSe2>(posMat));
        }
        else if (typeStr == "lm2d")
        {
            const DATA_TYPE x = *reinterpret_cast<DATA_TYPE*>(ptr);
            ptr += STEP_8;
            const DATA_TYPE y = *reinterpret_cast<DATA_TYPE*>(ptr);
            ptr += STEP_8;

            graph->AddVertex(id, std::make_unique<Vertex2d>(Eigen::Vector2f(x, y)));
        }
    }
    const auto edgesCount = *reinterpret_cast<unsigned*>(ptr);
    std::cout << "Edges count: " << edgesCount << std::endl;
    ptr += STEP_4;

    for (unsigned i = 0; i < edgesCount; ++i)
    {
        const unsigned typeStrSize = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;
        std::string typeStr {ptr, ptr + typeStrSize};
        ptr += typeStrSize;
        const unsigned id1 = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;
        const unsigned id2 = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;
        const unsigned rows = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;
        const unsigned cols = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;

        DynamicMatrix<float> measurement;
        DynamicMatrix<float> information;
        if (rows == 0)
        {
            if (typeStr == "se2")
            {
                const DATA_TYPE x = *reinterpret_cast<DATA_TYPE*>(ptr);
                ptr += STEP_8;
                const DATA_TYPE y = *reinterpret_cast<DATA_TYPE*>(ptr);
                ptr += STEP_8;
                const DATA_TYPE theta = *reinterpret_cast<DATA_TYPE*>(ptr);
                ptr += STEP_8;
                measurement = CreateTransform2d(x, y, theta);
            }
            else if (typeStr == "se2point2")
            {
                measurement = Eigen::Vector2f(*reinterpret_cast<DATA_TYPE*>(ptr), 
                                                *reinterpret_cast<DATA_TYPE*>(ptr + STEP_8));
                ptr += STEP_8 * 2;
            }
            else
            {
                throw "NOT IMPLE INF edge type";
            }
        }
        else
        {
            if (typeStr == "se2")
            {
                measurement = Eigen::Matrix3f::Identity();

                for (int r = 0; r < cols; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        measurement(r, c) = *reinterpret_cast<DATA_TYPE*>(ptr);
                        ptr += STEP_8;
                    }
                }
            }
            else
            {
                throw "NOT IMPLE INF edge type";
            }
        }

        const unsigned rowsInf = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;
        const unsigned colsInf = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;

        if (rowsInf == 0)
        {
            if (typeStr == "se2")
            {
                information = Eigen::Matrix3f::Identity();
            }
            else if (typeStr == "se2point2")
            {
                information = Eigen::Matrix2f::Identity();
            }
            else
            {
                throw "NOT IMPLE INF edge type";
            }

            for (int j = 0; j < colsInf; ++j)
            {
                information(j, j) = *reinterpret_cast<DATA_TYPE*>(ptr);
                ptr += STEP_8;
            }
        }
        else
        {
            throw "NOT IMPLE INF NxN";
        }
        
        if (typeStr == "se2")
        {
            graph->AddEdge(std::make_unique<EdgeSE2>(id1, id2, measurement, information));
        }
        else if (typeStr == "se2point2")
        {
            graph->AddEdge(std::make_unique<EdgeSE2Point2d>(id1, id2, measurement, information));
        }
        else
        {
            throw "NOT IMPLE INF edge type";
        }
    }

    const auto fixedVerticesCount = *reinterpret_cast<unsigned*>(ptr);
    std::cout << "Vertices count: " << verticesCount << std::endl;
    ptr += STEP_4;

    for (int i = 0; i < fixedVerticesCount; ++i)
    {
        const auto id = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;

        graph->FixVertex(id);
    }

    return graph;
}