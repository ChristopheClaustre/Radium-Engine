#include "NeighborsSelectionWithRegularGrid.hpp"

#include <APSS/NeighborsSelection/RegularGrid/RegularGridBuilder.hpp>
#include <APSS/NeighborsSelection/RegularGrid/RegularGrid.hpp>

namespace PointyCloudPlugin {

NeighborsSelectionWithRegularGrid::NeighborsSelectionWithRegularGrid(std::shared_ptr<PointyCloud> cloud, float influenceRadius, int nCell) :
    NeighborsSelection(cloud, influenceRadius),
    m_grid(RegularGridBuilder::buildRegularGrid(cloud, nCell))
{
}

NeighborsSelectionWithRegularGrid::~NeighborsSelectionWithRegularGrid()
{
}

void NeighborsSelectionWithRegularGrid::getNeighbors(const APoint &point, std::vector<int> & indexSelected) const
{
    m_grid->query(point.pos(), m_influenceRadius, indexSelected);
}


void NeighborsSelectionWithRegularGrid::processNeighbors(const APoint& point, NeighborsProcessor& f) const
{
    m_grid->process(point.pos(), m_influenceRadius, f);
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
