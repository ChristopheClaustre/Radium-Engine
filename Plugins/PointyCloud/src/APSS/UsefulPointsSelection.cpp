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
    Ra::Core::Vector3 view = m_camera->getDirection();
    PointyCloud pc(*m_originalCloud.get());

    auto pointsFirst = pc.m_points.begin();
    auto pointsEnd = pc.m_points.end();
    for(auto pointsIt = pc.m_points.begin(); pointsIt!=pointsEnd; ++pointsIt)
    {
        if(view.dot(pointsIt->normal()) < 0)
            *pointsFirst++ = *pointsIt;
    }

    pc.m_points.erase(pointsFirst, pointsEnd);

    if (pc.m_points.size() == 0)
        pc.m_points.push_back(m_originalCloud->m_points[0]);

    return pc;
}

} // namespace PointyCloudPlugin












