#ifndef POINTYCLOUDPLUGIN_UPSAMPLERSIMPLE_H
#define POINTYCLOUDPLUGIN_UPSAMPLERSIMPLE_H

#include <APSS/UpSampler/UpSampler.hpp>

#include <Engine/Renderer/Camera/Camera.hpp>

namespace PointyCloudPlugin {

class UpSamplerSimple:public UpSampler
{
public:
    UpSamplerSimple(Scalar threshold, const Ra::Engine::Camera& camera);
    virtual ~UpSamplerSimple();
    virtual void upSampleCloud(const PointyCloud& usefulPointCloud, int N);

    void setThreshold(int ts) { m_threshold = ts; }

protected:

    float m_threshold;
    const Ra::Engine::Camera &m_camera;

    virtual inline int getM(const APoint& point);
    Scalar computeEta(const APoint& point);

    inline Ra::Core::Vector2 project(const Ra::Core::Vector3& p) const;
    inline Ra::Core::Vector3 pointInView(const Ra::Core::Vector3& p) const;
}; // class UpSamplerSimple

} // namespace PointyCloudPlugin

#include <APSS/UpSampler/UpSamplerSimple.inl>

#endif // POINTYCLOUDPLUGIN_UPSAMPLERSIMPLE_H
