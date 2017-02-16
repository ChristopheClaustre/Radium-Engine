#include "OrthogonalProjection.hpp"
//#include "PointyCloud.hpp"
#include "NeighborsSelection.hpp"


namespace PointyCloudPlugin {

OrthogonalProjection::OrthogonalProjection(std::shared_ptr<NeighborsSelection> neighborsSelection,
                                           std::shared_ptr<PointyCloud> originalCloud,
                                            double influenceRadius) :
                                           Scalar influenceRadius) :
    m_selector(neighborsSelection),
    m_originalCloud(originalCloud),
    m_influenceRadius(influenceRadius)
{
}

OrthogonalProjection::~OrthogonalProjection()
{
}

void OrthogonalProjection::project(PointyCloud &upSampledCloud)
{
    Fit fit;
    fit.setWeightFunc(WeightFunc(m_influenceRadius));

    for(auto &p : upSampledCloud.m_points)
    {
        fit.init(p.pos());

        std::vector<int> neighbors = m_selector->getNeighbors(p);

        for(auto &idx : neighbors)
            fit.addNeighbor(m_originalCloud->m_points[idx]);

        int i = 0;
        while(fit.finalize()==Grenaille::NEED_OTHER_PASS && ++i<MAX_FITTING_ITERATION)
            for(auto &idx : neighbors)
                fit.addNeighbor(m_originalCloud->m_points[idx]);

        p.pos() = fit.project(p.pos());
        p.normal() = fit.primitiveGradient(p.pos());
    }
}

} // namespace PointyCloudPlugin
