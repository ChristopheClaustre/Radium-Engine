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
    int imin = fmax(floor((double)(q[0]-r)/grid.m_dx), 0.0);
    int jmin = fmax(floor((double)(q[1]-r)/grid.m_dy), 0.0);
    int kmin = fmax(floor((double)(q[2]-r)/grid.m_dz), 0.0);
    int imax = fmin(floor((double)(q[0]+r)/grid.m_dx), grid.m_nx-1.0);
    int jmax = fmin(floor((double)(q[1]+r)/grid.m_dy), grid.m_ny-1.0);
    int kmax = fmin(floor((double)(q[2]+r)/grid.m_dz), grid.m_nz-1.0);

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
                        fun(grid.m_indices[idx]);
            }
}

template<typename F> __device__
void processNeighborsAll(const Vector3& p, Scalar r, int size, const Vector3* positions, F& fun)
{
    for (int idx = 0; idx < size; ++idx)
        if( (p - positions[idx]).norm() <=r )
            fun(idx);
}

} // namespace Cuda
} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_NEIGHBORS_H
