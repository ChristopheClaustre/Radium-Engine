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
    //TODO pass it as parameter?
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

    // initialize cells
    grid->m_cells.resize(grid->m_nx*grid->m_ny*grid->m_nz, RegularGrid::Cell());

    // grid filling
    auto begin = grid->m_indices.begin();
    for(int k = 0; k < grid->m_indices.size();++k)
    {
        // corresponding cell
        int idxCell = grid->rawIndex(cloud.m_points[k].pos());

        // index in m_indices
        int pos = grid->m_cells[idxCell].index + grid->m_cells[idxCell].length + 1;

        // shift elements such that index k is located at pos
        std::rotate(begin+pos, begin+k,begin+k+1);

        // update current cell (increment length)
        ++grid->m_cells[idxCell].length;

        // increment all next cells index
        for(auto cellIt = grid->m_cells.begin()+idxCell+1; cellIt!=grid->m_cells.end(); ++cellIt)
            ++(cellIt->index);
    }

    return grid;
}

Ra::Core::Aabb RegularGridBuilder::computeAabb(const PointyCloud& cloud)
{
    Ra::Core::Aabb aabb;

    for(auto& p : cloud.m_points)
        aabb.extend(p.pos());

    // add an extra space at max corner
    const float epsilon = 1e-5;
    Ra::Core::Vector3 e;
    e << epsilon, epsilon, epsilon;
    aabb.extend(aabb.max()+e);

    return aabb;
}

} // namespace PointyCloudPlugin
