#ifndef POINTYCLOUDPLUGIN_NEIGHBORS_H
#define POINTYCLOUDPLUGIN_NEIGHBORS_H

#include <Cuda/defines.h>
#include <Cuda/RegularGrid.h>

namespace PointyCloudPlugin {
namespace Cuda {

template<typename F> __device__
void processNeighbors(const Vector3& p, Scalar r, const RegularGrid& grid, const Vector3* positions, F& fun)
{
    // point in local coordinates
    Vector3 q = p - grid.m_aabb.min();

    // searching limits of a cube centered at q of length r
    int imin = fmax(floor((q[0]-r)/grid.m_dx), 0);
    int jmin = fmax(floor((q[1]-r)/grid.m_dy), 0);
    int kmin = fmax(floor((q[2]-r)/grid.m_dz), 0);
    int imax = fmin(floor((q[0]+r)/grid.m_dx), grid.m_nx-1);
    int jmax = fmin(floor((q[1]+r)/grid.m_dy), grid.m_ny-1);
    int kmax = fmin(floor((q[2]+r)/grid.m_dz), grid.m_nz-1);

    // search
    for(int k = kmin; k<=kmax; ++k)
        for(int j = jmin; j<=jmax; ++j)
            for(int i = imin; i<=imax; ++i)
            {
                int idxCell = grid.rawIndex(i, j, k);
                int begin = grid.m_cells[idxCell].begin;
                int length = grid.m_cells[idxCell].length;

                for(int idx = begin; idx<begin+length; ++idx)
                    if((p - positions[grid.m_indices[idx]]).norm() <= r)
                        fun(idx);
            }
}

} // namespace Cuda
} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_NEIGHBORS_H
