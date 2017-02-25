#ifndef POINTYCLOUDPLUGIN_CUDA_REGULARGRID_H
#define POINTYCLOUDPLUGIN_CUDA_REGULARGRID_H

#include <Cuda/defines.h>

#include <Eigen/Core>

namespace PointyCloudPlugin {
namespace Cuda {

struct RegularGrid
{
public:
    //TODO use vector of defines.h !
    typedef float Scalar;
    typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
    typedef Eigen::Matrix<Scalar, 4, 1> Vector4;

    struct Cell {
        int index;
        int size;
    };

    RegularGrid(size_t size, const Vector3* positions);
    ~RegularGrid();

    // device array
    int*  m_indices;
    Cell* m_cells;

}; // class RegularGrid

} // namespace Cuda
} // namespace PointyCloudPlugin


#endif // POINTYCLOUDPLUGIN_CUDA_REGULARGRID_H
