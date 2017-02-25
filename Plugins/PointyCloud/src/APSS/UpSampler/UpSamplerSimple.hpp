#ifndef POINTYCLOUDPLUGIN_UPSAMPLERSIMPLE_H
#define POINTYCLOUDPLUGIN_UPSAMPLERSIMPLE_H

#include <APSS/UpSampler/UpSampler.hpp>

#include <Engine/Renderer/Camera/Camera.hpp>

namespace PointyCloudPlugin {

class UpSamplerSimple:public UpSampler
{
public:
    UpSamplerSimple(float rayon, float threshold, const Ra::Engine::Camera& camera);
    ~UpSamplerSimple();
    virtual void upSampleCloud(PointyCloud& cloud);

    void setThreshold(int ts) { m_threshold = ts; }

protected:

    float m_threshold;
    const Ra::Engine::Camera &m_camera;

    int getM(const int& indice);
    int calculEta(const int& indice);

    inline Ra::Core::Vector2 project(const Ra::Core::Vector3& p) const
    {
        Ra::Core::Vector4 point = Ra::Core::Vector4::Ones();
        point.head<3>() = p;
        Ra::Core::Vector4 vpPoint = m_camera.getProjMatrix() * point;
        return Ra::Core::Vector2( m_camera.getWidth()  * 0.5f * (vpPoint.x()/vpPoint.z() + 1),
                              m_camera.getHeight() * 0.5f *  (1 - vpPoint.y()/vpPoint.z()));
    }

    inline Ra::Core::Vector3 pointInView(const Ra::Core::Vector3& p) const
    {
        Ra::Core::Vector4 point = Ra::Core::Vector4::Ones();
        point.head<3>() = p;
        Ra::Core::Vector4 vpPoint = m_camera.getViewMatrix() * point;
        return Ra::Core::Vector3(vpPoint.x()/vpPoint.w() ,vpPoint.y() / vpPoint.w(),vpPoint.z() / vpPoint.w());

    }
}; // class UpSamplerSimple

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_UPSAMPLERSIMPLE_H
