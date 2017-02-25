#ifndef POINTYCLOUDPLUGIN_TEST_H
#define POINTYCLOUDPLUGIN_TEST_H

#include <Cuda/defines.h>
#include <Cuda/APSS.h>

namespace PointyCloudPlugin {
namespace Cuda {

// just a kernel test that copy original data to final one
// no blocks or thread
// just change the color from white to red!

__global__
void copy(size_t sizeOriginal, Vector3* posIn,  Vector3* norIn,  Vector4* colIn,
          size_t sizeFinal,    Vector3* posOut, Vector3* norOut, Vector4* colOut, Scalar* splatSizeOut)
{
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    int stride = blockDim.x * gridDim.x;

    for (int i = 0; i < sizeOriginal; ++i)
    {
        float t = (float)i/(sizeOriginal-1);

        posOut[i] = posIn[i];
        norOut[i] = norIn[i];
        colOut[i][0] = 1;
        colOut[i][1] = t;
        colOut[i][2] = t;
        colOut[i][3] = 1;
        splatSizeOut[i] = 1.0;
    }
}


// just a kernel test that copy selected data to final one
// no blocks or thread
// just change the color from white to red!

__global__
void copySelected(size_t sizeSelected, Vector3* posIn,  Vector3* norIn,  Vector4* colIn,
                  int* selected, Vector3* posOut, Vector3* norOut, Vector4* colOut, Scalar* splatSizeOut,
                  Scalar splatRadius)
{
    for (int i = 0; i < sizeSelected; ++i)
    {
        float t = (float)i/(sizeSelected-1);

        posOut[i] = posIn[selected[i]];
        norOut[i] = norIn[selected[i]];
        colOut[i][0] = 1;
        colOut[i][1] = t;
        colOut[i][2] = t;
        colOut[i][3] = 1;
        splatSizeOut[i] = splatRadius;
    }
}

} // namespace Cuda
} // namespace PointyCloudPlugin


#endif // POINTYCLOUDPLUGIN_TEST_H
