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

} // namespace PointyCloudPlugin
