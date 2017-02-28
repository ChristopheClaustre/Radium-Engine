#ifndef POINTYCLOUDPLUGIN_SELECTIONKERNEL_H
#define POINTYCLOUDPLUGIN_SELECTIONKERNEL_H

#include <Cuda/defines.h>

namespace PointyCloudPlugin {
namespace Cuda {

__global__
void checkVisibility(size_t size, Vector3* positions, Vector3* normals, Vector3 cameraPosition, Vector3 cameraDirection, int* visibility)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<size)
        visibility[i] = (cameraDirection.dot(normals[i]) < 0);
}

__global__
void selectVisible(size_t size, int* visibility, int* visibilitySum, int* selected)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<size && visibility[i]==1)
            selected[visibilitySum[i]] = i;
}

} // namespace Cuda
} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_SELECTIONKERNEL_H
