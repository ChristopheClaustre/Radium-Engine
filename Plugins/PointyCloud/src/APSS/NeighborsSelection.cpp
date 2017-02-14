#include "NeighborsSelection.hpp"

namespace PointyCloudPlugin
{

NeighborsSelection::NeighborsSelection(std::shared_ptr<Ra::Engine::Mesh> cloud,const float influenceRadius) :
    m_cloud(cloud),
    m_influenceRadius(influenceRadius)
{
}

NeighborsSelection::~NeighborsSelection()
{
}

std::vector<int>  NeighborsSelection::getNeighbors(Ra::Core::Vector3 vertexPosition)
{
   std::vector<int> indexSelected;
   auto verticesIt = m_cloud->getGeometry().m_vertices.begin();
   auto beginIt = verticesIt;

   while(verticesIt != m_cloud->getGeometry().m_vertices.end())
   {
       if((*verticesIt - vertexPosition).norm() >= m_influenceRadius)
       {
           indexSelected.push_back(verticesIt-beginIt);
       }
   }
   return indexSelected;
}

} // namespace PointyCloudPlugin
