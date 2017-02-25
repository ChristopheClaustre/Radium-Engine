#include <Cuda/APSS.h>

#include <Cuda/Test.h> // <- to delete
#include <Cuda/SelectionKernel.h>
#include <Cuda/UpsamplingKernel.h>
#include <Cuda/ProjectionKernel.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <iostream>

namespace PointyCloudPlugin {
namespace Cuda {

APSS::APSS(const Vector3* positions,
           const Vector3* normals,
           const Vector4* colors,
           size_t size) :
    m_grid()
{
    m_grid = new RegularGrid(size, positions);

    // device allocation
    m_sizeOriginal = size;
    CUDA_ASSERT( cudaMalloc(&m_positionOriginal, size*sizeof(Vector3)) );
    CUDA_ASSERT( cudaMalloc(&m_normalOriginal,   size*sizeof(Vector3)) );
    CUDA_ASSERT( cudaMalloc(&m_colorOriginal,    size*sizeof(Vector4)) );

    CUDA_ASSERT( cudaMalloc(&m_visibility,    size*sizeof(int)) );
    CUDA_ASSERT( cudaMalloc(&m_visibilitySum, size*sizeof(int)) );
    CUDA_ASSERT( cudaMalloc(&m_selected,      size*sizeof(int)) );

    //TEST for test only
    // sizeFinal depends on generated splats count!
    m_sizeFinal = size;
    CUDA_ASSERT( cudaMalloc(&m_positionFinal,  size*sizeof(Vector3)) );
    CUDA_ASSERT( cudaMalloc(&m_normalFinal,    size*sizeof(Vector3)) );
    CUDA_ASSERT( cudaMalloc(&m_colorFinal,     size*sizeof(Vector4)) );
    CUDA_ASSERT( cudaMalloc(&m_splatSizeFinal, size*sizeof(Scalar)) );
    m_positionFinalHost  = new Vector3[size];
    m_normalFinalHost    = new Vector3[size];
    m_colorFinalHost     = new Vector4[size];
    m_splatSizeFinalHost = new Scalar[size];

    // other allocations ...
    // regular grid initialization ...
    // set eligibility ...

    // device transfert
    CUDA_ASSERT( cudaMemcpy(m_positionOriginal, positions, size*sizeof(Vector3), cudaMemcpyHostToDevice) );
    CUDA_ASSERT( cudaMemcpy(m_normalOriginal,   normals,   size*sizeof(Vector3), cudaMemcpyHostToDevice) );
    CUDA_ASSERT( cudaMemcpy(m_colorOriginal,    colors,    size*sizeof(Vector4), cudaMemcpyHostToDevice) );
}

APSS::~APSS()
{
    // device desallocation
    // ...
}

void APSS::select(const Vector3 &cameraPosition, const Vector3 &cameraDirection)
{
    checkVisibility<<<1,1>>>(m_sizeOriginal, m_positionOriginal, m_normalOriginal, cameraPosition, cameraDirection, m_visibility);
    CUDA_ASSERT( cudaPeekAtLastError() );
    CUDA_ASSERT( cudaDeviceSynchronize() );

    // prefix sum
    thrust::device_ptr<int> devPtr = thrust::device_pointer_cast(m_visibility);
    thrust::device_ptr<int> devPtrSum = thrust::device_pointer_cast(m_visibilitySum);
    thrust::exclusive_scan(thrust::device, devPtr, devPtr+m_sizeOriginal, devPtrSum, 0);

    updateSelectedCount();

    selectVisible<<<1,1>>>(m_sizeOriginal, m_visibility, m_visibilitySum, m_selected);
    CUDA_ASSERT( cudaPeekAtLastError() );
    CUDA_ASSERT( cudaDeviceSynchronize() );
}

void APSS::upsample(/*APSS parameters*/)
{
    //TODO
//    kernel<<</*numBlocks, blockSize*/>>>(...);
//    cudaDeviceSynchronize();
}

void APSS::project(/*APSS parameters*/)
{
    //TEST : copy original in final
//    copy<<</*numBlocks, blockSize*/1,1>>>(m_sizeOriginal, m_positionOriginal, m_normalOriginal, m_colorOriginal,
//                                          m_sizeFinal,    m_positionFinal,    m_normalFinal,    m_colorFinal, m_splatSizeFinal);

    copySelected<<<1,1>>>(m_sizeSelected, m_positionOriginal, m_normalOriginal, m_colorOriginal,
                 m_selected, m_positionFinal, m_normalFinal, m_colorFinal, m_splatSizeFinal);

    m_sizeFinal = m_sizeSelected;

    CUDA_ASSERT( cudaPeekAtLastError() );
    CUDA_ASSERT( cudaDeviceSynchronize() );
}

void APSS::finalize()
{
    // get back final data from device to host
    CUDA_ASSERT( cudaMemcpy(m_positionFinalHost,  m_positionFinal, m_sizeFinal*sizeof(Vector3), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(m_normalFinalHost,    m_normalFinal, m_sizeFinal*sizeof(Vector3), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(m_colorFinalHost,     m_colorFinal, m_sizeFinal*sizeof(Vector4), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(m_splatSizeFinalHost, m_splatSizeFinal, m_sizeFinal*sizeof(Scalar), cudaMemcpyDeviceToHost) );
}

void APSS::updateSelectedCount()
{
    m_sizeSelected = -1;
    CUDA_ASSERT( cudaMemcpy(&m_sizeSelected, m_visibilitySum+m_sizeOriginal-1, sizeof(int), cudaMemcpyDeviceToHost) );
}


} // namespace Cuda

} // namespace PointyCloudPlugin
