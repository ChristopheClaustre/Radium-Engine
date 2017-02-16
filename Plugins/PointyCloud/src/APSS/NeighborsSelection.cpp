#include "NeighborsSelection.hpp"

namespace PointyCloudPlugin
{

NeighborsSelection::NeighborsSelection(std::shared_ptr<PointyCloud> cloud, const float influenceRadius) :
    m_cloud(cloud),
    m_influenceRadius(influenceRadius)
{
}

NeighborsSelection::~NeighborsSelection()
{
}

std::vector<int> NeighborsSelection::getNeighbors(APoint point)
{
   std::vector<int> indexSelected;
   auto beginIt = m_cloud->m_points.begin();

   for (auto currentIt = beginIt; currentIt != m_cloud->m_points.end(); ++currentIt)
   {
       if ((currentIt->pos() - point.pos()).norm() <= m_influenceRadius)
       {
           indexSelected.push_back(currentIt-beginIt);
       }
   }
   return indexSelected;
}

} // namespace PointyCloudPlugin
