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

void NeighborsSelection::getNeighbors(const APoint& point, std::vector<int> & indexSelected) const
{
    for(int i = 0; i < m_cloud->size(); ++i)
    {
        if ((m_cloud->at(i).pos() - point.pos()).norm() <= m_influenceRadius)
        {
            indexSelected.push_back(i);
        }
    }
}

bool NeighborsSelection::isEligible(const APoint& point) const
{
   int neighbors = 0;
   int i = 0;
   while (i < m_cloud->size() && neighbors < 6)
   {
       if ((m_cloud->at(i).pos() - point.pos()).norm() <= m_influenceRadius)
       {
           ++neighbors;
       }
       ++i;
   }
   return (neighbors >=6);
}

} // namespace PointyCloudPlugin
