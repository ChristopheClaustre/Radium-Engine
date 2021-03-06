#ifndef POINTYCLOUDPLUGIN_ORTHOGONALPROJECTION_HPP
#define POINTYCLOUDPLUGIN_ORTHOGONALPROJECTION_HPP

#include <APSS/PointyCloud.hpp>
#include <PointyCloudPlugin.hpp>
#include <APSS/NeighborsSelection/NeighborsProcessor.hpp>
#include <grenaille.h>

#include <memory>

namespace PointyCloudPlugin {

#define MAX_FITTING_ITERATION 10
#define THRESHOLD_NORMAL 1 // degrees
#define THRESHOLD_POS    0.1 // % of splatRadius

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
    virtual ~OrthogonalProjection();

    inline void setInfluenceRadius(Scalar influenceRadius) { m_influenceRadius=influenceRadius; }

    void project(PointyCloud& upSampledCloud);

    // timing accessor
    Scalar getTimeNeighbors() const;
    Scalar getTimeFitting() const;
    Scalar getTimeProjecting() const;
    size_t getCount() const;
    size_t getMeanProjectionCount() const;

protected:

    std::shared_ptr<NeighborsSelection> m_selector;
    std::shared_ptr<PointyCloud> m_originalCloud;

    Scalar m_influenceRadius;

    // time stats
    Scalar m_timeNeighbors;
    Scalar m_timeFitting;
    Scalar m_timeProjecting;
    size_t m_count;
    size_t m_meanProjectionCount;
};

class AddFun : public NeighborsProcessor
{
public:
    AddFun(Fit* fit, const PointyCloud* cloud) :
        m_fit(fit), m_cloud(cloud) {}
    virtual ~AddFun(){}
    virtual inline void operator()(int idx) override {m_fit->addNeighbor(m_cloud->m_points[idx]);}

private:
    Fit* m_fit;
    const PointyCloud* m_cloud;
};


} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_ORTHOGONALPROJECTION_H
