#ifndef POINTYCLOUDPLUGIN_UPSAMPLINGKERNEL_H
#define POINTYCLOUDPLUGIN_UPSAMPLINGKERNEL_H

#include <Cuda/defines.h>

namespace PointyCloudPlugin {
namespace Cuda {

__global__
void computeSampleCountFixed(size_t sizeSelected, int m, int* splatCount)
{
    for (int i = 0; i < sizeSelected; ++i)
        splatCount[i] = m;
}

__global__
void generateSample(/*TODO*/)
{

}

} // namespace Cuda
} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_UPSAMPLINGKERNEL_H
