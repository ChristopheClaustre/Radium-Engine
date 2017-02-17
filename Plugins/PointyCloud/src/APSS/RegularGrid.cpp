#include "RegularGrid.hpp"

#include <Core/Log/Log.hpp>
#include <sstream>

namespace PointyCloudPlugin {

RegularGrid::RegularGrid()
{
}

RegularGrid::~RegularGrid()
{
}

std::vector<int> RegularGrid::query(const Ra::Core::Vector3& p, float r) const
{
    std::vector<int> indices;

    // point in local coordinates
    Ra::Core::Vector3 q = p - m_aabb.min();

    // searching limits of a cube centered at q of length r
    int imin = std::floor((q[0]-r)/m_dx);
    int imax = std::floor((q[0]+r)/m_dx);
    int jmin = std::floor((q[1]-r)/m_dx);
    int jmax = std::floor((q[1]+r)/m_dx);
    int kmin = std::floor((q[2]-r)/m_dx);
    int kmax = std::floor((q[2]+r)/m_dx);

    // clamp to grid
    imin = std::max(imin, 0);
    jmin = std::max(jmin, 0);
    kmin = std::max(kmin, 0);
    imax = std::min(imax, m_nx-1);
    jmax = std::min(jmax, m_ny-1);
    kmax = std::min(kmax, m_nz-1);

    // search
    for(int k = kmin; k<=kmax; ++k)
        for(int j = jmin; j<=jmax; ++j)
            for(int i = imin; i<=imax; ++i)
            {
                int idxCell = rawIndex(i, j, k);
                int begin = m_cells[idxCell].index;
                int length = m_cells[idxCell].length;

                // TODO : use insert, not push_back !
                for(int idx = begin; idx<begin+length; ++idx)
                    indices.push_back(m_indices[idx]);
            }

    return indices;
}

void RegularGrid::printAll() const
{
    std::stringstream output;
    output << "\nAabb = [" << m_aabb.min()[0] << "," << m_aabb.min()[1] << "," << m_aabb.min()[2] << "];[" <<
              m_aabb.max()[0] << "," << m_aabb.max()[1] << "," << m_aabb.max()[2] << "\n";
    output << "dx=" << m_dx << " " << "dy=" << m_dy << " " << "dz=" << m_dz << "\n";
    output << "nx=" << m_nx << " " << "ny=" << m_ny << " " << "nz=" << m_nz << "\n";

    LOG(logINFO) << "Regular grid :\n";
    LOG(logINFO) << output.str();

    printGrid();
}

void RegularGrid::printGrid() const
{
    std::stringstream output;
    int k = 0;
    for(const auto& cell : m_cells)
    {
        output << k++ <<":["<< cell.index <<","<< cell.length<<"] -> ";
        for(int i = cell.index; i < cell.index+cell.length; ++i)
            output << m_indices[i] << " ";
        output << "\n";
    }

    output << "\n";
    for(const auto& idx : m_indices)
        output << idx << " ";

    LOG(logINFO) << "\n" << output.str() << "\n";
}

} // namespace PointyCloudPlugin
