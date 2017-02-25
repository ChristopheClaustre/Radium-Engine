#ifndef POINTYCLOUDPLUGIN_DEFINES_H
#define POINTYCLOUDPLUGIN_DEFINES_H

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
    typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
    typedef Eigen::Matrix<Scalar, 4, 1> Vector4;




//    // device vector to pass in kernel
//    template<typename T>
//    struct KernelArray
//    {
//        MULTIARCH
//        KernelArray(thrust::device_vector<T>& dVec) :
//            _array(thrust::raw_pointer_cast(&dVec[0])), _size(dVec.size()) {}

//        MULTIARCH
//        ~KernelArray(){}

//        MULTIARCH
//        T& operator [](size_t idx) {return _array[idx];}

//        MULTIARCH
//        const T& operator [](size_t idx) const {return _array[idx];}

//        MULTIARCH
//        const size_t& size() const {return _size;}

//    private:
//        T*      _array;
//        size_t  _size;

//    }; // struct KernelArray

} // namespace Cuda
} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_DEFINES_H
