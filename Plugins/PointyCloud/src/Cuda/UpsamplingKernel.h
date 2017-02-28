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
    // orthonormal basis U,V normalIn
    Vector3 U;
    if(abs(normalIn[0])>1e-6)
        U = Vector3(-normalIn[1]/normalIn[0], 1, 0).normalized();
    else
        U = Vector3(1, 0, 0);
    Vector3 V = normalIn.cross(U).normalized();

    // generate (m x m) splats
    int m = sqrt((double)splatCount);
    double newSplatRadius = splatRadius/m;

    // "lower left" splat corner
    Vector3 corner = positionIn - splatRadius*(U+V);

    int k = 0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
        {
            positionOut[idxBegin+k]  = corner + (1+2*i)*newSplatRadius*U + (1+2*j)*newSplatRadius*V;
            normalOut[idxBegin+k]    = normalIn;
            colorOut[idxBegin+k]     = colorIn;
            splatSizeOut[idxBegin+k] = newSplatRadius;
            ++k;
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
