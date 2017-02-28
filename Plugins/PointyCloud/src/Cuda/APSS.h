#ifndef POINTYCLOUDPLUGIN_APSS_H
#define POINTYCLOUDPLUGIN_APSS_H

#include <Cuda/defines.h>
#include <Cuda/RegularGrid.h>

#include <Eigen/Core>

#define CUDA_THREADS 1024

namespace PointyCloudPlugin {
namespace Cuda {

class APSS
{
public:
    //TODO use vector of defines.h !
    typedef float Scalar;
    typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
    typedef Eigen::Matrix<Scalar, 4, 1> Vector4;

    APSS(const Vector3* positions,
         const Vector3* normals,
         const Vector4* colors,
         size_t size);
    ~APSS();

    // APSS steps
    void select(const Vector3& cameraPosition, const Vector3& cameraDirection);
    void upsample(int m, Scalar splatRadius);
    void project(Scalar influenceRadius);
    void finalize();

    // get APSS results
    inline const int&     sizeFinal() const {return m_sizeFinal;}
    inline const Vector3* positionFinal() const {return m_positionFinalHost;}
    inline const Vector3* normalFinal() const {return m_normalFinalHost;}
    inline const Vector4* colorFinal() const {return m_colorFinalHost;}
    inline const Scalar*  splatSizeFinal() const {return m_splatSizeFinalHost;}

private:

    void updateSelectedCount();
    void updateSampleCount();

    void updateFinalMemory();

    // regular grid for neighbors query
    RegularGrid* m_grid;

    //// device arrays

    // original cloud
    size_t   m_sizeOriginal;     // N
    Vector3* m_positionOriginal; // points
    Vector3* m_normalOriginal;
    Vector4* m_colorOriginal;
    bool*    m_eligible;

    // selection
    int  m_sizeSelected;   // M
    int* m_visibility;     // A
    int* m_visibilitySum;  // A'
    int* m_selected;       // V

    // upsampling
    int* m_splatCount;    // C
    int* m_splatCountSum; // C'

    // final cloud
    int      m_sizeFinal;     // P
    Vector3* m_positionFinal; // splats
    Vector3* m_normalFinal;
    Vector4* m_colorFinal;
    Scalar*  m_splatSizeFinal;

    //// host arrays

    // final cloud
    Vector3* m_positionFinalHost;
    Vector3* m_normalFinalHost;
    Vector4* m_colorFinalHost;
    Scalar*  m_splatSizeFinalHost;


}; // class APSS

} // namespace Cuda
} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_APSS_H