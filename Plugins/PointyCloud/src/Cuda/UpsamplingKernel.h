#ifndef POINTYCLOUDPLUGIN_UPSAMPLINGKERNEL_H
#define POINTYCLOUDPLUGIN_UPSAMPLINGKERNEL_H

#include <Cuda/defines.h>

namespace PointyCloudPlugin {
namespace Cuda {

__global__
void computeSampleCountFixed(int sizeSelected, int m, int* splatCount)
{
    for (int i = 0; i < sizeSelected; ++i)
        splatCount[i] = m;
}


__device__
void genSplat(int idxBegin, int splatCount, const Vector3& positionIn, const Vector3& normalIn, const Vector4& colorIn,
                                                  Vector3* positionOut, Vector3* normalOut, Vector4* colorOut)
{
    //TODO: Compute orthonormal basis U,V from normalIn
    const Vector3 U;
    const Vector3 V;

    for (int k = 0; k < splatCount; ++k)
    {
        //TODO: compute generated splat position
        /* /!\ k might be replace by i,j indices /?\ (cf upsampler)*/
        // for now the splats are at the same position
        positionOut[idxBegin+k] = positionIn;
        normalOut[idxBegin+k] = normalIn;
        colorOut[idxBegin+k] = colorIn;
    }
}


__global__
void generateSample(int sizeSelected, int* selected, int* splatCount, int* splatCountSum,
                    Vector3* positionOriginal, Vector3* normalOriginal, Vector4* colorOriginal,
                    Vector3* positionFinal,     Vector3* normalFinal,    Vector4* colorFinal)
{
    for (int i = 0; i < sizeSelected; ++i)
    {
        int idx = selected[i];
        genSplat(splatCountSum[i], splatCount[i], positionOriginal[idx], normalOriginal[idx], colorOriginal[idx],
                 positionFinal, normalFinal, colorFinal);
    }
}

} // namespace Cuda
} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_UPSAMPLINGKERNEL_H
