#ifndef POINTYCLOUDPLUGIN_CUDA_REGULARGRID_H
#define POINTYCLOUDPLUGIN_CUDA_REGULARGRID_H

#include <Cuda/defines.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace PointyCloudPlugin {
namespace Cuda {

struct RegularGrid
{
public:
    //TODO use vector of defines.h !
    typedef float Scalar;
    typedef Eigen::Matrix<Scalar, 3, 1>  Vector3;
    typedef Eigen::Matrix<Scalar, 4, 1>  Vector4;
    typedef Eigen::AlignedBox<Scalar, 3> Aabb;

    struct Cell {
        Cell() : begin(0), length(0) {}
        int begin;
        int length;
    };

    RegularGrid(size_t size, const Vector3* positions, int ncells = 80);
    ~RegularGrid();

    MULTIARCH
    inline int rawIndex(int i, int j, int k) const {
        return k*(m_nx*m_ny) + j*m_nx + i;
    }


    inline int rawIndex(const Vector3& p) const {
        return rawIndexLocal(p-m_aabb.min());
    }


    inline int rawIndexLocal(const Vector3& pLocal) const {
        return rawIndex(std::floor(pLocal[0]/m_dx),
                        std::floor(pLocal[1]/m_dy),
                        std::floor(pLocal[2]/m_dz));
    }

    void free();

//private:

    // bounding box
    Aabb m_aabb;

    // cells count
    int m_nx;
    int m_ny;
    int m_nz;

    // cells size
    Scalar m_dx;
    Scalar m_dy;
    Scalar m_dz;

    // device array
    int*  m_indices;
    Cell* m_cells;

}; // class RegularGrid

} // namespace Cuda
} // namespace PointyCloudPlugin


#endif // POINTYCLOUDPLUGIN_CUDA_REGULARGRID_H
