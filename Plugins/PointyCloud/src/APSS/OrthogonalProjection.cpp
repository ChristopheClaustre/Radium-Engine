#include "OrthogonalProjection.hpp"
#include "PointyCloud.hpp"
#include "NeighborsSelection.hpp"


namespace PointyCloudPlugin {

OrthogonalProjection::OrthogonalProjection(NeighborsSelection* neighborsSelection,
                     PointyCloud* originalCloud,
                     PointyCloud* upSampledCloud,
                     double influenceRadius) :
    m_selector(neighborsSelection),
    m_originalCloud(originalCloud),
    m_upSampledCloud(upSampledCloud),
    m_influenceRadius(influenceRadius)
{
}

OrthogonalProjection::~OrthogonalProjection()
{
}

void OrthogonalProjection::project()
{
    Fit fit;
    fit.setWeightFunc(WeightFunc(m_influenceRadius));

    for(auto &p : m_upSampledCloud->m_points)
    {
        fit.init(p.pos());

        std::vector<int> neighbors = m_selector->getNeighbors(p.pos());

        for(auto &idx : neighbors)
            fit.addNeighbor(m_originalCloud->m_points[idx]);

        int i = 0;

        while(fit.finalize()==Grenaille::NEED_OTHER_PASS && ++i<MAX_FITTING_ITERATION)
            for(auto &idx : neighbors)
                fit.addNeighbor(m_originalCloud->m_points[idx]);
    }
}

} // namespace PointyCloudPlugin
