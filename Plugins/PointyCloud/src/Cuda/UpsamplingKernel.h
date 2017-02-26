#ifndef POINTYCLOUDPLUGIN_UPSAMPLINGKERNEL_H
#define POINTYCLOUDPLUGIN_UPSAMPLINGKERNEL_H

#include <Cuda/defines.h>

namespace PointyCloudPlugin {
namespace Cuda {

__global__
void computeSampleCountFixed(int sizeSelected, int m, int* splatCount)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<sizeSelected)
        splatCount[i] = m*m;
}


__device__
void genSplat(int idxBegin, int splatCount, Scalar splatRadius,
              const Vector3& positionIn, const Vector3& normalIn, const Vector4& colorIn,
              Vector3* positionOut, Vector3* normalOut, Vector4* colorOut, Scalar* splatSizeOut)
{
    //TODO: Compute orthonormal basis U,V from normalIn
    const Vector3 U;
    const Vector3 V;

    for (int k = 0; k < splatCount; ++k)
    {
        //TODO: compute generated splat position
        /* /!\ k might be replaced by i,j indices /?\ (cf upsampler)*/
        // for now the splats are at the same position
        positionOut[idxBegin+k] = positionIn+ k*0.1*normalIn;
        normalOut[idxBegin+k] = normalIn;
        colorOut[idxBegin+k] = colorIn;
        splatSizeOut[idxBegin+k] = splatRadius;
    }
}

__global__
void generateSample(int sizeSelected, Scalar splatRadius, int* selected, int* splatCount, int* splatCountSum,
                    Vector3* positionOriginal, Vector3* normalOriginal, Vector4* colorOriginal,
                    Vector3* positionFinal,     Vector3* normalFinal,    Vector4* colorFinal, Scalar* splatSizeFinal)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<sizeSelected)
    {
        int idx = selected[i];
        genSplat(splatCountSum[i], splatCount[i], splatRadius,
                 positionOriginal[idx], normalOriginal[idx], colorOriginal[idx],
                 positionFinal, normalFinal, colorFinal, splatSizeFinal);
    }
}

} // namespace Cuda
} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_UPSAMPLINGKERNEL_H
