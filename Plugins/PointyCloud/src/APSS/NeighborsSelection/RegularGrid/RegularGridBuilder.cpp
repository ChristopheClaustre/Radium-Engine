#include "RegularGridBuilder.hpp"

#include <APSS/NeighborsSelection/RegularGrid/RegularGrid.hpp>
#include <APSS/PointyCloud.hpp>

#include <Core/Time/Timer.hpp>

#include <limits>

namespace PointyCloudPlugin {

std::unique_ptr<RegularGrid> RegularGridBuilder::buildRegularGrid(std::shared_ptr<PointyCloud> cloud, int nCell)
{
    Ra::Core::Timer::TimePoint start = Ra::Core::Timer::Clock::now();

    std::unique_ptr<RegularGrid> grid = std::make_unique<RegularGrid>();
    initialize(cloud, *grid.get(), nCell);
    fill(cloud, *grid.get());

    grid->m_buildTime = Ra::Core::Timer::getIntervalSeconds(start, Ra::Core::Timer::Clock::now());

    return grid;
}

void RegularGridBuilder::initialize(std::shared_ptr<PointyCloud> cloud,  RegularGrid& grid, int nCell)
{
    // set point cloud
    grid.m_cloud = cloud;

    // set bounding box
    grid.m_aabb = computeAabb(cloud);

    // fixed cells count along the 3 axis
    grid.m_nx = nCell;
    grid.m_ny = nCell;
    grid.m_nz = nCell;

    // cells size
    grid.m_dx = grid.m_aabb.diagonal()[0]/grid.m_nx;
    grid.m_dy = grid.m_aabb.diagonal()[1]/grid.m_ny;
    grid.m_dz = grid.m_aabb.diagonal()[2]/grid.m_nz;

    // initialize indices from 0 to size-1
    size_t size = cloud->size();
    grid.m_indices.resize(size);
    for(int idx = 0; idx<size; ++idx)
        grid.m_indices[idx] = idx;

    // initialize cells
    grid.m_cells.resize(grid.m_nx*grid.m_ny*grid.m_nz, RegularGrid::Cell());
}

void RegularGridBuilder::fill(std::shared_ptr<PointyCloud> cloud, RegularGrid &grid)
{
    auto begin = grid.m_indices.begin();
    #pragma omp parallel for
    for(int k = 0; k < grid.m_indices.size();++k)
    {
        // corresponding cell
        int idxCell = grid.rawIndex(cloud->at(k).pos());

        // index in m_indices
        int pos = grid.m_cells[idxCell].index + grid.m_cells[idxCell].length;

        // shift elements such that index k is located at pos
        std::rotate(begin+pos, begin+k,begin+k+1);

        // update current cell (increment length)
        ++grid.m_cells[idxCell].length;

        // increment all next cells index
        for(auto cellIt = grid.m_cells.begin()+idxCell+1; cellIt!=grid.m_cells.end(); ++cellIt)
            ++(cellIt->index);
    }
}

Ra::Core::Aabb RegularGridBuilder::computeAabb(std::shared_ptr<PointyCloud> cloud)
{
    Ra::Core::Aabb aabb;

    #pragma omp parallel for
    for(int i = 0; i < cloud->size(); ++i) {
        aabb.extend(cloud->at(i).pos());
    }

    // add an extra space at corners
    const float epsilon = 1e-5;
    Ra::Core::Vector3 e(epsilon, epsilon, epsilon);

    aabb.extend(aabb.max()+e);
    aabb.extend(aabb.min()-e);

    return aabb;
}

} // namespace PointyCloudPlugin
