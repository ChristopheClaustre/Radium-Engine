#include "UsefulPointsSelection.hpp"

namespace PointyCloudPlugin
{

UsefulPointsSelection::UsefulPointsSelection(const Ra::Engine::Mesh* cloud, const Ra::Engine::Camera* camera) :
    m_cloud(cloud),
    m_camera(camera)
{
}

UsefulPointsSelection::~UsefulPointsSelection()
{
}

std::vector<int> UsefulPointsSelection::selectUsefulPoints() const
{
    std::vector<int> useful(0);

    for(int idx = 0; idx < m_cloud->getGeometry().m_normals.size(); ++idx)
        if(isUsefulPoint(idx))
            useful.push_back(idx);

    return useful;
}

bool UsefulPointsSelection::isUsefulPoint(int idx) const
{
    Ra::Core::Vector3 normal = m_cloud->getGeometry().m_normals[idx];
    Ra::Core::Vector3 view = m_camera->getDirection();

    return view.dot(normal)<0;
}

} // namespace PointyCloudPlugin
