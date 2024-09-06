#pragma once

#include "serialization/DeserializeGraph.h"
#include "graph/Helper.h"
#include "tools/BlockTimer.h"

#include <iostream>

template<typename T, typename GraphT, typename Functions>
void DeserializeGraph(GraphT* graph, char* ptr) 
{
    if (graph == nullptr)
    {
        return;
    }
    BlockTimer timer{"DeserializeGraph"};

    const int STEP_4 = 4;
    const int STEP_T = sizeof(T);
    const auto verticesCount = *reinterpret_cast<unsigned*>(ptr);
    // std::cout << "Vertices count: " << verticesCount << std::endl;
    ptr += STEP_4;

    int globalLmId = 0;

    for (unsigned i = 0; i < verticesCount; ++i)
    {
        const unsigned id = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;
        VertexType type = static_cast<VertexType>(*reinterpret_cast<unsigned*>(ptr));
        ptr += STEP_4;
        if (type == VertexType::Se2)
        {
            const T x = *reinterpret_cast<T*>(ptr);
            ptr += STEP_T;
            const T y = *reinterpret_cast<T*>(ptr);
            ptr += STEP_T;
            const T theta = *reinterpret_cast<T*>(ptr);
            ptr += STEP_T;

            const auto posMat = CreateTransform2d<T>(x, y, theta);

            graph->AddVertex(id, Functions::CreateVertex(type, posMat));
        }
        else if (type == VertexType::Point2)
        {
            const T x = *reinterpret_cast<T*>(ptr);
            ptr += STEP_T;
            const T y = *reinterpret_cast<T*>(ptr);
            ptr += STEP_T;

            graph->AddVertex(id, Functions::CreateVertex(type, Eigen::Vector2<T>(x, y)));
        }
    }
    const auto edgesCount = *reinterpret_cast<unsigned*>(ptr);
    // std::cout << "Edges count: " << edgesCount << std::endl;
    ptr += STEP_4;

    for (unsigned i = 0; i < edgesCount; ++i)
    {
        EdgeType edgeType = static_cast<EdgeType>(*reinterpret_cast<unsigned*>(ptr));
        ptr += STEP_4;
        const unsigned id1 = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;
        const unsigned id2 = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;
        const unsigned rows = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;
        const unsigned cols = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;

        DynamicMatrix<T> measurement;
        DynamicMatrix<T> information;
        if (rows == 0)
        {
            if (edgeType == EdgeType::Se2)
            {
                const T x = *reinterpret_cast<T*>(ptr);
                ptr += STEP_T;
                const T y = *reinterpret_cast<T*>(ptr);
                ptr += STEP_T;
                const T theta = *reinterpret_cast<T*>(ptr);
                ptr += STEP_T;
                measurement = CreateTransform2d<T>(x, y, theta);
            }
            else if (edgeType == EdgeType::Se2Point2)
            {
                measurement = Eigen::Vector2<T>(*reinterpret_cast<T*>(ptr), 
                                                *reinterpret_cast<T*>(ptr + STEP_T));
                ptr += STEP_T * 2;
            }
            else
            {
                throw "NOT IMPLE INF edge type";
            }
        }
        else
        {
            if (edgeType == EdgeType::Se2)
            {
                measurement = Eigen::Matrix3<T>::Identity();

                for (int r = 0; r < cols; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        measurement(r, c) = *reinterpret_cast<T*>(ptr);
                        ptr += STEP_T;
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
            if (edgeType == EdgeType::Se2)
            {
                information = Eigen::Matrix3<T>::Identity();
            }
            else if (edgeType == EdgeType::Se2Point2)
            {
                information = Eigen::Matrix2<T>::Identity();
            }
            else
            {
                throw "NOT IMPLE INF edge type";
            }

            for (int j = 0; j < colsInf; ++j)
            {
                information(j, j) = *reinterpret_cast<T*>(ptr);
                ptr += STEP_T;
            }
        }
        else
        {
            throw "NOT IMPLE INF NxN";
        }
        
        if (edgeType == EdgeType::Se2)
        {
            graph->AddEdge(edgeType, Functions::CreateEdge(edgeType, id1, id2, measurement, information));
        }
        else if (edgeType == EdgeType::Se2Point2)
        {
            graph->AddEdge(edgeType, Functions::CreateEdge(edgeType, id1, id2, measurement, information));
        }
        else
        {
            throw "NOT IMPLE INF edge type";
        }
    }

    const auto fixedVerticesCount = *reinterpret_cast<unsigned*>(ptr);
    // std::cout << "Fixed vertices count: " << fixedVerticesCount << std::endl;
    ptr += STEP_4;

    for (int i = 0; i < fixedVerticesCount; ++i)
    {
        const auto id = *reinterpret_cast<unsigned*>(ptr);
        ptr += STEP_4;

        graph->FixVertex(id);
    }
}