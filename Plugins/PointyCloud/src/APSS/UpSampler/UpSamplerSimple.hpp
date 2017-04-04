#ifndef POINTYCLOUDPLUGIN_UPSAMPLERSIMPLE_H
#define POINTYCLOUDPLUGIN_UPSAMPLERSIMPLE_H

#include <APSS/UpSampler/UpSampler.hpp>

#include <Engine/Renderer/Camera/Camera.hpp>

namespace PointyCloudPlugin {

class UpSamplerSimple : public UpSampler
{
public:
    UpSamplerSimple(std::shared_ptr<PointyCloud> originalCloud,Scalar threshold, const Ra::Engine::Camera& camera);
    virtual ~UpSamplerSimple();

    void setThreshold(int ts) { m_threshold = ts; }

protected:
    void upSamplePointMaster(int index);

    virtual inline int getM(const APoint& point);
    Scalar computeEta(const APoint& point);

    inline Ra::Core::Vector2 project(const Ra::Core::Vector3& p) const;
    inline Ra::Core::Vector3 pointInView(const Ra::Core::Vector3& p) const;

protected:
    float m_threshold;
    const Ra::Engine::Camera &m_camera;

}; // class UpSamplerSimple

} // namespace PointyCloudPlugin

#include <APSS/UpSampler/UpSamplerSimple.inl>

#endif // POINTYCLOUDPLUGIN_UPSAMPLERSIMPLE_H
