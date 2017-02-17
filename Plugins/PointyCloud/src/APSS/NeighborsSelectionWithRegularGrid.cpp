#include "NeighborsSelectionWithRegularGrid.hpp"

#include "RegularGridBuilder.hpp"
#include "RegularGrid.hpp"

namespace PointyCloudPlugin {

NeighborsSelectionWithRegularGrid::NeighborsSelectionWithRegularGrid(std::shared_ptr<PointyCloud> cloud, float influenceRadius) :
    NeighborsSelection(cloud, influenceRadius),
    m_grid(RegularGridBuilder::buildRegularGrid(*cloud.get(), influenceRadius))
{
}

NeighborsSelectionWithRegularGrid::~NeighborsSelectionWithRegularGrid()
{
}

std::vector<int> NeighborsSelectionWithRegularGrid::getNeighbors(const APoint &point) const
{
    std::vector<int> indices = m_grid->query(point.pos(), m_influenceRadius);

    //TODO : remove indices far from more than radius

    return indices;
}

} // namespace PointyCloudPlugin
