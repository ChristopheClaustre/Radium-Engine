#include "UsefulPointsSelection.hpp"

#include <algorithm>

namespace PointyCloudPlugin
{

UsefulPointsSelection::UsefulPointsSelection(PointyCloud cloud, const Ra::Engine::Camera* camera) :
    m_cloud(cloud),
    m_camera(camera)
{
}

UsefulPointsSelection::~UsefulPointsSelection()
{
}

PointyCloud UsefulPointsSelection::selectUsefulPoints()
{
    Ra::Core::Vector3 view = m_camera->getDirection();
    PointyCloud pc = m_cloud;

    auto pointsFirst = pc.m_points.begin();
    auto pointsEnd = pc.m_points.end();
    for(auto pointsIt = pc.m_points.begin(); pointsIt!=pointsEnd; ++pointsIt)
    {
        if(view.dot(pointsIt->normal()) < 0)
            *pointsFirst++ = *pointsIt;
    }

    pc.m_points.erase(pointsFirst, pointsEnd);

    return pc;
}

} // namespace PointyCloudPlugin












