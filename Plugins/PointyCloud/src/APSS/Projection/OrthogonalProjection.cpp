#include "OrthogonalProjection.hpp"

#include <APSS/NeighborsSelection/NeighborsSelection.hpp>
#include <PointyCloudPlugin.hpp>

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
    ON_TIMED(
    m_count = 0;
    m_timeNeighbors = 0;
    m_timeFitting = 0;
    m_timeProjecting = 0;)
}

OrthogonalProjection::~OrthogonalProjection()
{
}

void OrthogonalProjection::project(PointyCloud &upSampledCloud)
{
    Fit fit;
    fit.setWeightFunc(WeightFunc(m_influenceRadius));

    ON_TIMED(
    Ra::Core::Timer::TimePoint start;
    float timeNeighbors = 0.0;
    float timeFitting = 0.0;
    float timeProjecting = 0.0;
    size_t pointToFitCount = 0;
    size_t projectionCount = 0;)

    #pragma omp parallel for
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

                ON_TIMED(start = Ra::Core::Timer::Clock::now();)
                neighbors.clear();
                m_selector->getNeighbors(p, neighbors);
                ON_TIMED(timeNeighbors += Ra::Core::Timer::getIntervalMicro(start, Ra::Core::Timer::Clock::now());)

                ON_TIMED(start = Ra::Core::Timer::Clock::now();)
                for(auto &idx : neighbors) {
                    fit.addNeighbor(m_originalCloud->m_points[idx]);
                }
                ON_TIMED(timeFitting += Ra::Core::Timer::getIntervalMicro(start, Ra::Core::Timer::Clock::now());)

                // As our fit is an OrientedSphereFit
                // finalize should never return NEED_OTHER_PASS
                // finalize should only return STABLE || UNSTABLE || UNDEFINED
                // we accept result in unstable state because its good enough ;)
                ON_TIMED(start = Ra::Core::Timer::Clock::now();)
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
                ON_TIMED(timeProjecting += Ra::Core::Timer::getIntervalMicro(start, Ra::Core::Timer::Clock::now());)

                ON_TIMED(++projectionCount;)
                i++;
            }

            ON_TIMED(++pointToFitCount;)
        }
    }

    // update timing attributes
    ON_TIMED(
    m_timeNeighbors += timeNeighbors/pointToFitCount;
    m_timeFitting += timeFitting/pointToFitCount;
    m_timeProjecting += timeProjecting/pointToFitCount;
    m_meanProjectionCount += projectionCount/pointToFitCount;
    ++m_count;)
}

ON_TIMED(
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
) // ON_TIMED end

} // namespace PointyCloudPlugin
