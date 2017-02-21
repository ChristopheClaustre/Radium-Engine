#include "NeighborsSelection.hpp"

namespace PointyCloudPlugin
{

NeighborsSelection::NeighborsSelection(std::shared_ptr<PointyCloud> cloud, const Scalar influenceRadius) :
    m_cloud(cloud),
    m_influenceRadius(influenceRadius)
{
}

NeighborsSelection::~NeighborsSelection()
{
}

std::vector<int> NeighborsSelection::getNeighbors(const APoint& point) const
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

bool NeighborsSelection::isEligible(const APoint& point) const
{
   int neighbors = 0;
   auto beginIt = m_cloud->m_points.begin();
   auto currentIt = beginIt;
   while (currentIt != m_cloud->m_points.end() && neighbors < 6)
   {
       if ((currentIt->pos() - point.pos()).norm() <= m_influenceRadius)
       {
           ++neighbors;
       }
       ++currentIt;
   }
   return (neighbors >=6);
}

} // namespace PointyCloudPlugin
