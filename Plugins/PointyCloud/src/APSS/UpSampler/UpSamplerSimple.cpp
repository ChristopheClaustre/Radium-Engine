#include "UpSamplerSimple.hpp"

namespace  PointyCloudPlugin {

UpSamplerSimple::UpSamplerSimple(std::shared_ptr<PointyCloud> originalCloud, Scalar threshold, const Ra::Engine::Camera & camera)
    : UpSampler(originalCloud), m_threshold(threshold), m_camera(camera)
{
}

UpSamplerSimple::~UpSamplerSimple()
{
}

void UpSamplerSimple::upSamplePointMaster(int index)
{
    const APoint& point = m_originalCloud->at(index);
    this->upSamplePoint(getM(point), point, index);
}

// return the number of pixel that takes an splat of radius 1 at the position of the index_th pixel
Scalar UpSamplerSimple::computeEta(const APoint& point)
{
    Scalar skewFactor;
    // m_point are already normalized
    if(m_camera.getProjType() == Ra::Engine::Camera::ProjType::ORTHOGRAPHIC)
    {
        skewFactor = m_camera.getDirection().dot(point.normal());
    }
    else
    {
        Ra::Core::Vector3 distPToCam = point.pos() - m_camera.getPosition();
        skewFactor = distPToCam.normalized().dot(point.normal());
    }
    skewFactor = std::max(skewFactor+1,Scalar(0));

    const Ra::Core::Vector3 originalPointInView = pointInView(point.pos());
    const Ra::Core::Vector3 extremPoint =
            (originalPointInView + Ra::Core::Vector3( 1.0, 0.0, 0.0 ));

    const Ra::Core::Vector2 extremPointProj = project(extremPoint);
    const Ra::Core::Vector2 originalPointProj = project(originalPointInView);

    Ra::Core::Vector2 diff = originalPointProj - extremPointProj;
    diff[0] *= m_camera.getWidth();
    diff[1] *= m_camera.getHeight();
    Scalar diameterInProj = diff.norm() * 2;

    return skewFactor * diameterInProj * diameterInProj;

    // exact compute should be : area of splat * skew factor
    // but this is enough for our use and easier to compute
}

} // namespace PointyCloudPlugin
