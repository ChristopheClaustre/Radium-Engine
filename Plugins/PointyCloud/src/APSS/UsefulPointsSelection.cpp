#include "UsefulPointsSelection.hpp"

#include <algorithm>

namespace PointyCloudPlugin
{

UsefulPointsSelection::UsefulPointsSelection(std::shared_ptr<Ra::Engine::Mesh> cloud, const Ra::Engine::Camera* camera) :
    m_cloud(cloud),
    m_camera(camera)
{
}

UsefulPointsSelection::~UsefulPointsSelection()
{
}

void UsefulPointsSelection::selectUsefulPoints()
{
    Ra::Core::Vector3 view = m_camera->getDirection();

    auto normalFirst = m_cloud->getGeometry().m_normals.begin();
    auto vertexFirst = m_cloud->getGeometry().m_vertices.begin();
    auto colorFirst  = m_cloud->getData(Ra::Engine::Mesh::VERTEX_COLOR).begin();

    auto normalLast = m_cloud->getGeometry().m_normals.end();

    auto normalIt = m_cloud->getGeometry().m_normals.begin();
    auto vertexIt = m_cloud->getGeometry().m_vertices.begin();
    auto colorIt  = m_cloud->getData(Ra::Engine::Mesh::VERTEX_COLOR).begin();

    size_t newSize = 0;

    for(; normalIt!=normalLast; ++normalIt, ++vertexIt, ++colorIt)
    {
        if(view.dot(*normalIt)<0)
        {
            *normalFirst++ = *normalIt;
            *vertexFirst++ = *vertexIt;
            *colorFirst++  = *colorIt;
            ++newSize;
        }
    }

    m_cloud->getGeometry().m_normals.resize(newSize);
    m_cloud->getGeometry().m_vertices.resize(newSize);
    m_cloud->getData(Ra::Engine::Mesh::VERTEX_COLOR).resize(newSize);

}

} // namespace PointyCloudPlugin












