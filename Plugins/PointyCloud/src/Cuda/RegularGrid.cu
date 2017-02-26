#include <Cuda/RegularGrid.h>

#include <vector>

namespace PointyCloudPlugin {
namespace Cuda {

RegularGrid::RegularGrid(size_t size, const Vector3 *positions, int ncells)
{
    // bounding box
    Aabb aabb;
    for(int k = 0; k<size; ++k)
        aabb.extend(positions[k]);

    // add extra space at corners
    const Scalar epsilon = 1e-5;
    Vector3 e(epsilon, epsilon, epsilon);
    aabb.extend(aabb.max()+e);
    aabb.extend(aabb.min()-e);

    // fixed cells count along the 3 axis
    m_nx = ncells;
    m_ny = ncells;
    m_nz = ncells;

    // cells size
    m_dx = m_aabb.diagonal()[0]/m_nx;
    m_dy = m_aabb.diagonal()[1]/m_ny;
    m_dz = m_aabb.diagonal()[2]/m_nz;

    // initialize indices
    std::vector<int> indices(size);
    for(int idx = 0; idx<size; ++idx)
        indices[idx] = idx;

    // initialize cells
    std::vector<Cell> cells(m_nx*m_ny*m_nz, Cell());

    // fill
    std::vector<int>::iterator begin = indices.begin();
    for(int k = 0; k < indices.size();++k)
    {
        // corresponding cell
        int idxCell = rawIndex(positions[k]);

        // index in m_indices
        int pos = cells[idxCell].begin + cells[idxCell].length;

        // shift elements such that index k is located at pos
        // TODO: shift in the other sens?
        std::rotate(begin+pos, begin+k,begin+k+1);

        // update current cell (increment length)
        ++cells[idxCell].length;

        // increment all next cells index
        for(std::vector<Cell>::iterator cellIt = cells.begin()+idxCell+1; cellIt!=cells.end(); ++cellIt)
            ++(cellIt->begin);
    }

    // allocate device memory
    CUDA_ASSERT( cudaMalloc(&m_indices, size*sizeof(int)) );
    CUDA_ASSERT( cudaMalloc(&m_cells,   size*sizeof(Cell)) );

    // send data
    CUDA_ASSERT( cudaMemcpy(m_indices, indices.data(), size*sizeof(int) ,cudaMemcpyHostToDevice) );
    CUDA_ASSERT( cudaMemcpy(m_cells,   cells.data(),   size*sizeof(Cell),cudaMemcpyHostToDevice) );
}

RegularGrid::~RegularGrid()
{
}

void RegularGrid::free()
{
    // free device memory
    CUDA_ASSERT( cudaFree(m_indices) );
    CUDA_ASSERT( cudaFree(m_cells) );
}


} // namespace Cuda
} // namespace PointyCloudPlugin
