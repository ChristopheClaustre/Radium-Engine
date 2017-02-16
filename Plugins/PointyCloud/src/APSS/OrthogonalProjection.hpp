#ifndef POINTYCLOUDPLUGIN_ORTHOGONALPROJECTION_H
#define POINTYCLOUDPLUGIN_ORTHOGONALPROJECTION_H

#include <grenaille.h>

#include <memory>

namespace PointyCloudPlugin {

class APoint;

#define MAX_FITTING_ITERATION 10

// Define related structure
typedef Grenaille::DistWeightFunc<APoint, Grenaille::SmoothWeightKernel<double> > WeightFunc;
typedef Grenaille::Basket<APoint, WeightFunc, Grenaille::AlgebraicSphere, Grenaille::OrientedSphereFit> Fit;

class PointyCloud;
class NeighborsSelection;

class OrthogonalProjection
{
public:
    OrthogonalProjection(std::shared_ptr<NeighborsSelection> neighborsSelection,
                         std::shared_ptr<PointyCloud> originalCloud,
                         double influenceRadius);
    ~OrthogonalProjection();

    inline void setInfluenceRadius(double influenceRadius){m_influenceRadius=influenceRadius;}

    void project(PointyCloud& upSampledCloud);

protected:

    std::shared_ptr<NeighborsSelection> m_selector;
    std::shared_ptr<PointyCloud> m_originalCloud;

    double m_influenceRadius;
};



} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_ORTHOGONALPROJECTION_H
