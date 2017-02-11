#include "UsefulPointsSelection.hpp"

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

    auto normalIt   = m_cloud->getGeometry().m_normals.begin();
    auto verticesIt = m_cloud->getGeometry().m_vertices.begin();
    auto colorsIt   = m_cloud->getData(Ra::Engine::Mesh::VERTEX_COLOR).begin();

    while(normalIt != m_cloud->getGeometry().m_normals.end())
    {
        if(view.dot(*normalIt)>0)
        {
            normalIt = m_cloud->getGeometry().m_normals.erase(normalIt);
            verticesIt = m_cloud->getGeometry().m_vertices.erase(verticesIt);
            colorsIt = m_cloud->getData(Ra::Engine::Mesh::VERTEX_COLOR).erase(colorsIt);
        }
        else
        {
            ++normalIt;
            ++verticesIt;
            ++colorsIt;
        }
    }
}

} // namespace PointyCloudPlugin
