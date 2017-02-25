#include "OrthogonalProjection.hpp"

#include <APSS/NeighborsSelection/NeighborsSelection.hpp>

#include <Core/Time/Timer.hpp>

namespace PointyCloudPlugin {

OrthogonalProjection::OrthogonalProjection(std::shared_ptr<NeighborsSelection> neighborsSelection,
                                           std::shared_ptr<PointyCloud> originalCloud,
                                           Scalar influenceRadius) :
    m_selector(neighborsSelection),
    m_originalCloud(originalCloud),
    m_influenceRadius(influenceRadius)
{
    // timing
    m_count = 0;
    m_timeNeighbors = 0;
    m_timeFitting = 0;
    m_timeProjecting = 0;
}

OrthogonalProjection::~OrthogonalProjection()
{
}

void OrthogonalProjection::project(PointyCloud &upSampledCloud)
{
    Fit fit;
    fit.setWeightFunc(WeightFunc(m_influenceRadius));


    for(int i = 0; i < upSampledCloud.m_points.size(); ++i)
    {
        auto &p = upSampledCloud.m_points[i];
        if (p.eligible())
        {
            float threshold = 0.001 * p.radius();
            float diff = p.radius();
            int i = 0;
            std::vector<int> neighbors;
            while(diff >= threshold && i < MAX_FITTING_ITERATION)
            {
                fit.init(p.pos());

                neighbors.clear();
                m_selector->getNeighbors(p, neighbors);

                for(auto &idx : neighbors) {
                    fit.addNeighbor(m_originalCloud->m_points[idx]);
                }
                // As our fit is an OrientedSphereFit
                // finalize should never return NEED_OTHER_PASS
                // finalize should only return STABLE || UNSTABLE || UNDEFINED
                // we accept result in unstable state because its good enough ;)
                if (fit.finalize() != Grenaille::UNDEFINED) {
                    auto newPos = fit.project(p.pos());
                    auto newNormal = fit.primitiveGradient(newPos);
                    diff = (p.pos()-newPos).norm() + (p.normal()-newNormal).norm();
                    p.pos() = newPos;
                    p.normal() = newNormal;
                }
                else {
                    diff = 0;
                }
                i++;

        }
    }

// same project function for time recording
//void OrthogonalProjection::project(PointyCloud &upSampledCloud)
//{
//    Fit fit;
//    fit.setWeightFunc(WeightFunc(m_influenceRadius));

//    Ra::Core::Timer::TimePoint start;
//    float timeNeighbors = 0.0;
//    float timeFitting = 0.0;
//    float timeProjecting = 0.0;
//    size_t processedCount = 0;

//    for(auto &p : upSampledCloud.m_points)
//    {
//        if (p.isEligible())
//        {
//            fit.init(p.pos());

//            start = Ra::Core::Timer::Clock::now();
//            std::vector<int> neighbors = m_selector->getNeighbors(p);
//            timeNeighbors += Ra::Core::Timer::getIntervalMicro(start, Ra::Core::Timer::Clock::now());

//            start = Ra::Core::Timer::Clock::now();
//            int i = 0;
//            int res;
//            do
//            {
//                for(auto &idx : neighbors)
//                    fit.addNeighbor(m_originalCloud->m_points[idx]);

//                res = fit.finalize();
//                i++;
//            } while(res == Grenaille::NEED_OTHER_PASS && i<MAX_FITTING_ITERATION);

//            timeFitting += Ra::Core::Timer::getIntervalMicro(start, Ra::Core::Timer::Clock::now());

//            start = Ra::Core::Timer::Clock::now();
//            auto newPos = fit.project(p.pos());
//            APoint _p(newPos, fit.primitiveGradient(newPos), p.color());
//            p = _p;
//            timeProjecting += Ra::Core::Timer::getIntervalMicro(start, Ra::Core::Timer::Clock::now());

//            ++processedCount;
//        }
//    }

//    // update timing attributes
//    m_timeNeighbors += timeNeighbors/processedCount;
//    m_timeFitting += timeFitting/processedCount;
//    m_timeProjecting += timeProjecting/processedCount;
//    ++m_count;
//}

float OrthogonalProjection::getTimeNeighbors() const
{
    return m_count==0 ? 0.0 : m_timeNeighbors/m_count;
}

float OrthogonalProjection::getTimeFitting() const
{
    return m_count==0 ? 0.0 : m_timeFitting/m_count;
}

float OrthogonalProjection::getTimeProjecting() const
{
    return m_count==0 ? 0.0 : m_timeProjecting/m_count;
}

int OrthogonalProjection::getCount() const
{
    return m_count;
}

int OrthogonalProjection::getMeanProjectionCount() const
{
    return m_meanProjectionCount;
}
} // namespace PointyCloudPlugin
