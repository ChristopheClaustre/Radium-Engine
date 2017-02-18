#ifndef POINTYCLOUDPLUGIN_ORTHOGONALPROJECTION_HPP
#define POINTYCLOUDPLUGIN_ORTHOGONALPROJECTION_HPP

#include <grenaille.h>

#include <memory>

#include <APSS/PointyCloud.hpp>

namespace PointyCloudPlugin {

#define MAX_FITTING_ITERATION 10

// Define related structure
typedef Grenaille::DistWeightFunc<APoint, Grenaille::SmoothWeightKernel<Scalar> > WeightFunc;
typedef Grenaille::Basket<APoint, WeightFunc, Grenaille::AlgebraicSphere, Grenaille::OrientedSphereFit> Fit;

class PointyCloud;
class APoint;
class NeighborsSelection;

class OrthogonalProjection
{
public:
    OrthogonalProjection(std::shared_ptr<NeighborsSelection> neighborsSelection,
                         std::shared_ptr<PointyCloud> originalCloud,
                         Scalar influenceRadius);
    ~OrthogonalProjection();

    inline void setInfluenceRadius(Scalar influenceRadius) { m_influenceRadius=influenceRadius; }

    void project(PointyCloud& upSampledCloud);

protected:

    std::shared_ptr<NeighborsSelection> m_selector;
    std::shared_ptr<PointyCloud> m_originalCloud;

    Scalar m_influenceRadius;
};



} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_ORTHOGONALPROJECTION_H
