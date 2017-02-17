#include "RegularGridBuilder.hpp"

#include "RegularGrid.hpp"
#include "PointyCloud.hpp"

#include <limits>

namespace PointyCloudPlugin {

std::unique_ptr<RegularGrid> RegularGridBuilder::buildRegularGrid(const PointyCloud &cloud, double influenceRadius)
{
    std::unique_ptr<RegularGrid> grid = std::make_unique<RegularGrid>();    

    // set bounding box
    grid->m_aabb = computeAabb(cloud);

    // fixed cells count along the 3 axis
    grid->m_nx = 100;
    grid->m_ny = 100;
    grid->m_nz = 100;

    // cells size
    grid->m_dx = grid->m_aabb.diagonal()[0]/grid->m_nx;
    grid->m_dy = grid->m_aabb.diagonal()[1]/grid->m_ny;
    grid->m_dz = grid->m_aabb.diagonal()[2]/grid->m_nz;

    // initialize indices from 0 to size-1
    size_t size = cloud.m_points.size();
    grid->m_indices.resize(size);
    for(int idx = 0; idx<size; ++idx)
        grid->m_indices[idx] = idx;

    // initialize leaves
    grid->m_leaves.resize(grid->m_nx*grid->m_ny*grid->m_nz, RegularGrid::Cell());

    //TODO : fill grid

    return grid;
}

Ra::Core::Aabb RegularGridBuilder::computeAabb(const PointyCloud& cloud)
{
    Ra::Core::Aabb aabb;

    for(auto& p : cloud.m_points)
        aabb.extend(p.pos());

    return aabb;
}

} // namespace PointyCloudPlugin
