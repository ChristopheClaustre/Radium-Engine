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
    m_timeProjecting = 0;
    m_meanProjectionCount = 0;)
}

OrthogonalProjection::~OrthogonalProjection()
{
}

void OrthogonalProjection::project(PointyCloud &upSampledCloud)
{
#ifndef CORE_USE_OMP
    Fit fit;
    fit.setWeightFunc(WeightFunc(m_influenceRadius));
    AddFun addFun(&fit, m_originalCloud.get());
#endif

    Scalar threshold_n = cos(THRESHOLD_NORMAL * 180 / M_PI);
    Scalar initDiff_n = threshold_n*2;
    Scalar pourcentTs = THRESHOLD_POS/100;

    ON_TIMED(
    Ra::Core::Timer::TimePoint start;
    Scalar timeNeighbors =   0.0;
    Scalar timeFitting =     0.0;
    Scalar timeProjecting =  0.0;
    Scalar pointToFitCount = 0.0;
    size_t projectionCount = 0;)

#ifdef TIMED
    #pragma omp parallel for reduction(+:timeNeighbors,timeFitting,timeProjecting,pointToFitCount,projectionCount)
#else
    #pragma omp parallel for
#endif
    for(int i = 0; i < upSampledCloud.m_points.size(); ++i)
    {
#ifdef CORE_USE_OMP
        Fit fit;
        fit.setWeightFunc(WeightFunc(m_influenceRadius));
        AddFun addFun(&fit, m_originalCloud.get());
#endif

        auto &p = upSampledCloud.m_points[i];
        if (p.eligible())
        {
            Scalar diff_n = initDiff_n;

            Scalar threshold_p = pourcentTs * p.radius();
            Scalar diff_p = threshold_p*2;

            fit.init(p.pos());

            int i = 0;
            while((diff_n >= threshold_n || diff_p >= threshold_p) && i < MAX_FITTING_ITERATION)
            {
                ON_TIMED(start = Ra::Core::Timer::Clock::now();)
                m_selector->processNeighbors(p, addFun);
                ON_TIMED(timeNeighbors += Ra::Core::Timer::getIntervalMicro(start, Ra::Core::Timer::Clock::now());)
                ON_TIMED(timeFitting   += Ra::Core::Timer::getIntervalMicro(start, Ra::Core::Timer::Clock::now());)

                // As our fit is an OrientedSphereFit
                // finalize should never return NEED_OTHER_PASS
                // finalize should only return STABLE || UNSTABLE || UNDEFINED
                // we accept result in unstable state because its good enough ;)
                ON_TIMED(start = Ra::Core::Timer::Clock::now();)
                if (fit.finalize() == Grenaille::STABLE) {
                    auto newPos = fit.project(p.pos());
                    auto newNormal = fit.primitiveGradient(newPos);
                    diff_n = p.normal().dot(newNormal);
                    diff_p = (p.pos()-newPos).norm();
                    // update
                    p.pos() = newPos;
                    p.normal() = newNormal;
                }
                else {
                    diff_n = 0;
                    diff_p = 0;
                }
                ON_TIMED(timeProjecting += Ra::Core::Timer::getIntervalMicro(start, Ra::Core::Timer::Clock::now());)

                ++i;
            }
            ON_TIMED(
            projectionCount += i;
            ++pointToFitCount;)
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
Scalar OrthogonalProjection::getTimeNeighbors() const
{
    return m_count==0 ? 0.0 : m_timeNeighbors/m_count;
}

Scalar OrthogonalProjection::getTimeFitting() const
{
    return m_count==0 ? 0.0 : m_timeFitting/m_count;
}

Scalar OrthogonalProjection::getTimeProjecting() const
{
    return m_count==0 ? 0.0 : m_timeProjecting/m_count;
}

size_t OrthogonalProjection::getCount() const
{
    return m_count;
}

size_t OrthogonalProjection::getMeanProjectionCount() const
{
    return m_count==0 ? 0 : m_meanProjectionCount/m_count;
}
) // ON_TIMED end

} // namespace PointyCloudPlugin
