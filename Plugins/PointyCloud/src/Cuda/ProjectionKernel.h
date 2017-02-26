#ifndef POINTYCLOUDPLUGIN_PROJECTIONKERNEL_H
#define POINTYCLOUDPLUGIN_PROJECTIONKERNEL_H

#include <Cuda/defines.h>
#include <Cuda/Neighbors.h>
#include <Cuda/RegularGrid.h>

#include <grenaille.h>

namespace PointyCloudPlugin {
namespace Cuda {

struct Point {
    typedef float Scalar;
    typedef Vector3 VectorType;

    MULTIARCH inline Point(VectorType* positions, VectorType* normals, int idx) :
        m_pos   (&positions[idx]),
        m_normal(&normals[idx]  ) {}

    MULTIARCH inline const VectorType& pos()    const { return *m_pos; }
    MULTIARCH inline const VectorType& normal() const { return *m_normal; }

private:
    const VectorType* m_pos;
    const VectorType* m_normal;
};

typedef Grenaille::DistWeightFunc<Point, Grenaille::SmoothWeightKernel<Scalar> > WeightFunc;
typedef Grenaille::Basket<Point, WeightFunc, Grenaille::AlgebraicSphere, Grenaille::OrientedSphereFit> Fit;

struct AddNeighborsFunctor
{
    MULTIARCH inline AddNeighborsFunctor(Fit* fit, Vector3* positions, Vector3* normals) : fit_(fit), positions_(positions), normals_(normals) {}

    MULTIARCH inline void operator() (int idx) {fit_->addNeighbor(Point(positions_, normals_, idx));}

    Fit*     fit_;
    Vector3* positions_;
    Vector3* normals_;
};

__global__
void projection(Vector3* positionsOriginal, Vector3* normalsOriginal, RegularGrid grid, Scalar influenceRadius,
             int sizeFinal, Vector3* positionsFinal, Vector3* normalsFinal)
{
    Fit fit;
    fit.setWeightFunc(WeightFunc(influenceRadius));

    AddNeighborsFunctor functor(&fit, positionsOriginal, normalsOriginal);

    for(int k = 0; k<sizeFinal; ++k)
    {
        fit.init(positionsFinal[k]);

        processNeighbors(positionsFinal[k], influenceRadius, grid, positionsOriginal, functor);

        fit.finalize();

        positionsFinal[k] = fit.project(positionsFinal[k]);
        normalsFinal[k]   = fit.primitiveGradient(positionsFinal[k]);
    }
}

} // namespace Cuda
} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_PROJECTIONKERNEL_H
