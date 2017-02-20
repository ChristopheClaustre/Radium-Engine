#include "NeighborsSelectionWithRegularGrid.hpp"

#include "RegularGridBuilder.hpp"
#include "RegularGrid.hpp"

namespace PointyCloudPlugin {

NeighborsSelectionWithRegularGrid::NeighborsSelectionWithRegularGrid(std::shared_ptr<PointyCloud> cloud, float influenceRadius, int nCell) :
    NeighborsSelection(cloud, influenceRadius),
    m_grid(RegularGridBuilder::buildRegularGrid(cloud, nCell))
{
}

NeighborsSelectionWithRegularGrid::~NeighborsSelectionWithRegularGrid()
{
}

std::vector<int> NeighborsSelectionWithRegularGrid::getNeighbors(const APoint &point) const
{
    return m_grid->query(point.pos(), m_influenceRadius);
}

bool NeighborsSelectionWithRegularGrid::isEligible(const APoint& point) const
{
    return m_grid->hasNeighbors(point.pos(), m_influenceRadius);
}

const RegularGrid* NeighborsSelectionWithRegularGrid::grid() const
{
    return m_grid.get();
}


} // namespace PointyCloudPlugin
