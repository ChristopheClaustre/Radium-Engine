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

    // searching limits
    int imin = std::floor((q[0]-r)/m_dx);
    int imax = std::floor((q[0]+r)/m_dx);
    int jmin = std::floor((q[1]-r)/m_dx);
    int jmax = std::floor((q[1]+r)/m_dx);
    int kmin = std::floor((q[2]-r)/m_dx);
    int kmax = std::floor((q[2]+r)/m_dx);

    for(int i = imin; i<=imax; ++i)
        for(int j = jmin; j<=jmax; ++j)
            for(int k = kmin; k<=kmax; ++k)
            {
                int idx = rawIndex(i, j, k);
                //TODO fill indices with corresponding leave at idx
            }

    return indices;
}

} // namespace PointyCloudPlugin
