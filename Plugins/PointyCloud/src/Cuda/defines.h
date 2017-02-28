#ifndef POINTYCLOUDPLUGIN_DEFINES_H
#define POINTYCLOUDPLUGIN_DEFINES_H

#include <cuda_runtime.h>
#include <driver_types.h>

#include <stdio.h>
#include <Eigen/Core>

#define MULTIARCH __host__ __device__

#define CUDA_ASSERT(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace PointyCloudPlugin {
namespace Cuda {

    typedef float Scalar;
    typedef Eigen::Matrix<Scalar, 3, 1>  Vector3;
    typedef Eigen::Matrix<Scalar, 4, 1>  Vector4;
    typedef Eigen::AlignedBox<Scalar, 3> Aabb;

} // namespace Cuda
} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_DEFINES_H
