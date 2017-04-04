#include "NeighborsSelection.hpp"

#include <APSS/NeighborsSelection/NeighborsProcessor.hpp>

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

void NeighborsSelection::processNeighbors(const APoint& point, NeighborsProcessor& f) const
{
    auto beginIt = m_cloud->m_points.begin();

    for (auto currentIt = beginIt; currentIt != m_cloud->m_points.end(); ++currentIt)
    {
        if ((currentIt->pos() - point.pos()).norm() <= m_influenceRadius)
        {
            f(currentIt-beginIt);
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
