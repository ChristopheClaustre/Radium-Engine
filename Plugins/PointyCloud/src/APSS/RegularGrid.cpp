#include "RegularGrid.hpp"

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
    kmax = std::min(jmax, m_nz-1);

    // search
    for(int k = kmin; k<=kmax; ++k)
        for(int j = jmin; j<=jmax; ++j)
            for(int i = imin; i<=imax; ++i)
            {
                int idxLeave = rawIndex(i, j, k);
                int begin = m_leaves[idxLeave].index;
                int length = m_leaves[idxLeave].length;

                // TODO : use insert, not push_back
                for(int idx = begin; idx<begin+length; ++idx)
                    indices.push_back(m_indices[idx]);
            }

    return indices;
}

} // namespace PointyCloudPlugin
