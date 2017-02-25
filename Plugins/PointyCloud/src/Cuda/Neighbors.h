#ifndef POINTYCLOUDPLUGIN_NEIGHBORS_H
#define POINTYCLOUDPLUGIN_NEIGHBORS_H

#include <Cuda/defines.h>
#include <Cuda/RegularGrid.h>

namespace PointyCloudPlugin {
namespace Cuda {

template<typename F> __device__
void processNeighbors(const Vector3& p, Scalar r, const RegularGrid& grid, F& fun)
{
    //TODO
}

} // namespace Cuda
} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_NEIGHBORS_H
