#include "OrthogonalProjection.hpp"
#include "NeighborsSelection.hpp"


namespace PointyCloudPlugin {

OrthogonalProjection::OrthogonalProjection(std::shared_ptr<NeighborsSelection> neighborsSelection,
                                           std::shared_ptr<PointyCloud> originalCloud,
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

        int i = 0;
        int res;
        do
        {
            for(auto &idx : neighbors)
                fit.addNeighbor(m_originalCloud->m_points[idx]);

            res = fit.finalize();
//            std::cout << "finalize() -> " << res << std::endl << std::flush;
            i++;
        } while(res!=Grenaille::STABLE/*Grenaille::NEED_OTHER_PASS*/ && i<MAX_FITTING_ITERATION);

//        std::cout << "nombre de fitting : " << i << std::endl << std::flush;
//        std::cout << "avant : " << p.pos().transpose() << " après : " << fit.project(p.pos()).transpose() << std::endl << std::flush;

        APoint _p(fit.project(p.pos()), fit.primitiveGradient(p.pos()), p.color());
        p = _p;
    }
}

} // namespace PointyCloudPlugin
