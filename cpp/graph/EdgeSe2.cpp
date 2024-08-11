#include "graph/EdgeSe2.h"

#include "graph/Graph.h"
#include "graph/Helper.h"

std::tuple<DynamicVector<float>, DynamicMatrix<float>, DynamicMatrix<float>> 
            EdgeSE2::CalcErrorAndJ(const Graph<float>* graph)
{
    const auto& pos_1 = graph->GetVertex(id1)->position;
    const auto& pos_2 = graph->GetVertex(id2)->position;

    auto pp = pos_1.inverse() * pos_2;
    auto delta = meas.inverse() * pp;
    auto err = Eigen::Vector3f{ delta(0, 2), delta(1, 2), ConvertMatToAngle(delta) };

    // std::cout << "SE2: " << id1 << " - " << id2 << ", err2: " << err[2] << "\n";

    const auto A = -Eigen::Matrix3f::Identity();
    const auto B = Eigen::Matrix3f::Identity();

    return {err, A, B};
}