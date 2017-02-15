#ifndef POINTYCLOUDPLUGIN_ORTHOGONALPROJECTION_H
#define POINTYCLOUDPLUGIN_ORTHOGONALPROJECTION_H

#include <grenaille.h>

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
    OrthogonalProjection(NeighborsSelection* neighborsSelection,
                         PointyCloud* originalCloud,
                         PointyCloud* upSampledCloud,
                         double influenceRadius);
    ~OrthogonalProjection();

    inline void setInfluenceRadius(double influenceRadius){m_influenceRadius=influenceRadius;}

    void project();

protected:

    NeighborsSelection* m_selector;
    PointyCloud* m_originalCloud;
    PointyCloud* m_upSampledCloud;

    double m_influenceRadius;
};



} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_ORTHOGONALPROJECTION_H
