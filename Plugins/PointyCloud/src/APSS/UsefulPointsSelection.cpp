#include "UsefulPointsSelection.hpp"

#include <algorithm>

namespace PointyCloudPlugin
{

UsefulPointsSelection::UsefulPointsSelection(std::shared_ptr<PointyCloud> originalCloud, const Ra::Engine::Camera* camera) :
    m_originalCloud(originalCloud),
    m_camera(camera)
{
}

UsefulPointsSelection::~UsefulPointsSelection()
{
}

PointyCloud UsefulPointsSelection::selectUsefulPoints()
{
    // check visibility from camera
    PointyCloud visiblePoints(*m_originalCloud.get());
    selectFromVisibility(visiblePoints);

    // this method must return at least 1 point
    if (visiblePoints.m_points.size() == 0)
        visiblePoints.m_points.push_back(m_originalCloud->m_points[0]);

    // check orientation
    PointyCloud wellOrientedPoints(visiblePoints);
    selectFromOrientation(wellOrientedPoints);

    // camera must be inside the cloud
    // if it is the case we must return the opposite face of the "mesh"
    //   (the not well oriented one (all the visible in fact))
    if (wellOrientedPoints.m_points.size() == 0)
        return visiblePoints;

    return wellOrientedPoints;
}

inline void UsefulPointsSelection::selectFromVisibility(PointyCloud &pc)
{
    Ra::Core::Matrix4 VP = m_camera->getProjMatrix() * m_camera->getViewMatrix();

    auto pointsFirst = pc.m_points.begin();
    auto pointsEnd = pc.m_points.end();
    for(auto pointsIt = pc.m_points.begin(); pointsIt!=pointsEnd; ++pointsIt)
    {
        Ra::Core::Vector4 point = Ra::Core::Vector4::Ones();
        point.head<3>() = pointsIt->pos();
        auto vpPoint = VP * point;

        auto X = vpPoint[0] / vpPoint[3];
        auto Y = vpPoint[1] / vpPoint[3];
        auto Z = vpPoint[2] / vpPoint[3];
        if (X <= 1 && Y <= 1 && Z <= 1 && X >= -1 && Y >= -1 && Z >= -1)
        {
            *pointsFirst++ = *pointsIt;
        }
    }

    pc.m_points.erase(pointsFirst, pointsEnd);
}

inline void UsefulPointsSelection::selectFromOrientation(PointyCloud &pc)
{
    Ra::Core::Vector3 view = m_camera->getDirection();

    auto pointsFirst = pc.m_points.begin();
    auto pointsEnd = pc.m_points.end();
    for(auto pointsIt = pc.m_points.begin(); pointsIt!=pointsEnd; ++pointsIt)
    {
        if(view.dot(pointsIt->normal()) < 0)
            *pointsFirst++ = *pointsIt;
    }

    pc.m_points.erase(pointsFirst, pointsEnd);
}

} // namespace PointyCloudPlugin












