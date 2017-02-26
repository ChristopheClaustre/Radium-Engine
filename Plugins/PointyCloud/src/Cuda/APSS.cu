#include <Cuda/APSS.h>

#include <Cuda/Test.h> // TODO: delete this include
#include <Cuda/SelectionKernel.h>
#include <Cuda/UpsamplingKernel.h>
#include <Cuda/ProjectionKernel.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

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
    CUDA_ASSERT( cudaMalloc(&m_eligible,         size*sizeof(bool)) );

    CUDA_ASSERT( cudaMalloc(&m_visibility,    size*sizeof(int)) );
    CUDA_ASSERT( cudaMalloc(&m_visibilitySum, size*sizeof(int)) );
    CUDA_ASSERT( cudaMalloc(&m_selected,      size*sizeof(int)) );
    CUDA_ASSERT( cudaMalloc(&m_splatCount,    size*sizeof(int)) );
    CUDA_ASSERT( cudaMalloc(&m_splatCountSum, size*sizeof(int)) );

    // allocation of final data device memory to arbitrary size
    m_sizeFinal = size;
    CUDA_ASSERT( cudaMalloc(&m_positionFinal,  size*sizeof(Vector3)) );
    CUDA_ASSERT( cudaMalloc(&m_normalFinal,    size*sizeof(Vector3)) );
    CUDA_ASSERT( cudaMalloc(&m_colorFinal,     size*sizeof(Vector4)) );
    CUDA_ASSERT( cudaMalloc(&m_splatSizeFinal, size*sizeof(Scalar)) );

    // allocation of final data host memory to arbitrary size
    m_positionFinalHost  = new Vector3[size];
    m_normalFinalHost    = new Vector3[size];
    m_colorFinalHost     = new Vector4[size];
    m_splatSizeFinalHost = new Scalar[size];

    //TODO: set eligibility ...

    // device transfert
    CUDA_ASSERT( cudaMemcpy(m_positionOriginal, positions, size*sizeof(Vector3), cudaMemcpyHostToDevice) );
    CUDA_ASSERT( cudaMemcpy(m_normalOriginal,   normals,   size*sizeof(Vector3), cudaMemcpyHostToDevice) );
    CUDA_ASSERT( cudaMemcpy(m_colorOriginal,    colors,    size*sizeof(Vector4), cudaMemcpyHostToDevice) );
}

APSS::~APSS()
{
    CUDA_ASSERT( cudaFree(m_positionOriginal) );
    CUDA_ASSERT( cudaFree(m_normalOriginal) );
    CUDA_ASSERT( cudaFree(m_colorOriginal) );
    CUDA_ASSERT( cudaFree(m_eligible) );

    CUDA_ASSERT( cudaFree(m_visibility) );
    CUDA_ASSERT( cudaFree(m_visibilitySum) );
    CUDA_ASSERT( cudaFree(m_selected) );

    CUDA_ASSERT( cudaFree(m_splatCount) );
    CUDA_ASSERT( cudaFree(m_splatCountSum) );

    CUDA_ASSERT( cudaFree(m_positionFinal) );
    CUDA_ASSERT( cudaFree(m_normalFinal) );
    CUDA_ASSERT( cudaFree(m_colorFinal) );
    CUDA_ASSERT( cudaFree(m_splatSizeFinal) );

    free(m_positionFinalHost);
    free(m_normalFinalHost);
    free(m_colorFinalHost);
    free(m_splatSizeFinalHost);

    m_grid->free();
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

void APSS::upsample(int m, Scalar splatRadius)
{
    computeSampleCountFixed<<<1,1>>>(m_sizeSelected, m, m_splatCount);
    CUDA_ASSERT( cudaPeekAtLastError() );
    CUDA_ASSERT( cudaDeviceSynchronize() );

    // prefix sum
    thrust::device_ptr<int> devPtr = thrust::device_pointer_cast(m_splatCount);
    thrust::device_ptr<int> devPtrSum = thrust::device_pointer_cast(m_splatCountSum);
    thrust::exclusive_scan(thrust::device, devPtr, devPtr+m_sizeSelected, devPtrSum, 0);

    updateSampleCount();
    updateFinalMemory();

    generateSample<<<1,1>>>(m_sizeSelected, splatRadius, m_selected, m_splatCount, m_splatCountSum,
                            m_positionOriginal, m_normalOriginal, m_colorOriginal,
                            m_positionFinal, m_normalFinal, m_colorFinal, m_splatSizeFinal);
    CUDA_ASSERT( cudaPeekAtLastError() );
    CUDA_ASSERT( cudaDeviceSynchronize() );
}

void APSS::project(Scalar influenceRadius)
{
    projection<<<1,1>>>(m_sizeOriginal, m_positionOriginal, m_normalOriginal, *m_grid, influenceRadius,
                        m_sizeFinal,    m_positionFinal,    m_normalFinal);

    CUDA_ASSERT( cudaPeekAtLastError() );
    CUDA_ASSERT( cudaDeviceSynchronize() );
}

void APSS::finalize()
{
    // get back final data from device to host
    CUDA_ASSERT( cudaMemcpy(m_positionFinalHost,  m_positionFinal,  m_sizeFinal*sizeof(Vector3), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(m_normalFinalHost,    m_normalFinal,    m_sizeFinal*sizeof(Vector3), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(m_colorFinalHost,     m_colorFinal,     m_sizeFinal*sizeof(Vector4), cudaMemcpyDeviceToHost) );
    CUDA_ASSERT( cudaMemcpy(m_splatSizeFinalHost, m_splatSizeFinal, m_sizeFinal*sizeof(Scalar),  cudaMemcpyDeviceToHost) );
}

void APSS::updateSelectedCount()
{
    m_sizeSelected = -1;
    CUDA_ASSERT( cudaMemcpy(&m_sizeSelected, m_visibilitySum+m_sizeOriginal-1, sizeof(int), cudaMemcpyDeviceToHost) );
}

void APSS::updateSampleCount()
{
    m_sizeFinal = -1;
    CUDA_ASSERT( cudaMemcpy(&m_sizeFinal, m_splatCountSum+m_sizeSelected-1, sizeof(int), cudaMemcpyDeviceToHost) );
}

void APSS::updateFinalMemory()
{
    // free device memory
    CUDA_ASSERT( cudaFree(m_positionFinal) );
    CUDA_ASSERT( cudaFree(m_normalFinal) );
    CUDA_ASSERT( cudaFree(m_colorFinal) );
    CUDA_ASSERT( cudaFree(m_splatSizeFinal) );

    // alloc device memory
    CUDA_ASSERT( cudaMalloc(&m_positionFinal,  m_sizeFinal*sizeof(Vector3)) );
    CUDA_ASSERT( cudaMalloc(&m_normalFinal,    m_sizeFinal*sizeof(Vector3)) );
    CUDA_ASSERT( cudaMalloc(&m_colorFinal,     m_sizeFinal*sizeof(Vector4)) );
    CUDA_ASSERT( cudaMalloc(&m_splatSizeFinal, m_sizeFinal*sizeof(Scalar)) );

    // realloc host memory
    m_positionFinalHost  = (Vector3*)realloc(m_positionFinalHost,  m_sizeFinal*sizeof(Vector3));
    m_normalFinalHost    = (Vector3*)realloc(m_normalFinalHost,    m_sizeFinal*sizeof(Vector3));
    m_colorFinalHost     = (Vector4*)realloc(m_colorFinalHost,     m_sizeFinal*sizeof(Vector4));
    m_splatSizeFinalHost = (Scalar*) realloc(m_splatSizeFinalHost, m_sizeFinal*sizeof(Scalar));
}

} // namespace Cuda

} // namespace PointyCloudPlugin
