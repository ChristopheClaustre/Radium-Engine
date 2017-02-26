#include "UpSamplerSimple.hpp"

namespace PointyCloudPlugin {

inline int UpSamplerSimple::getM(const int& index)
{
    return round(sqrt(computeEta(index) * m_cloud->m_points[index].radius()/ m_threshold));
    // sqrt because this result gives mxm
}

// project a point already defined from the camera position to the perspective view
// in [0;1] for all dimensions
inline Ra::Core::Vector2 UpSamplerSimple::project(const Ra::Core::Vector3& p) const
{
    Ra::Core::Vector4 point = Ra::Core::Vector4::Ones();
    point.head<3>() = p;
    Ra::Core::Vector4 vpPoint = m_camera.getProjMatrix() * point;
    return Ra::Core::Vector2(
                std::max(std::min(0.5f * (vpPoint.x()/vpPoint.w() + 1), Scalar(1)), Scalar(0)),
                std::max(std::min(0.5f * (1 - vpPoint.y()/vpPoint.w()), Scalar(1)), Scalar(0)));
}

// redefined a point from the camera position
inline Ra::Core::Vector3 UpSamplerSimple::pointInView(const Ra::Core::Vector3& p) const
{
    Ra::Core::Vector4 point = Ra::Core::Vector4::Ones();
    point.head<3>() = p;
    Ra::Core::Vector4 vpPoint = m_camera.getViewMatrix() * point;
    return Ra::Core::Vector3(vpPoint.x()/vpPoint.w() ,vpPoint.y() / vpPoint.w(),vpPoint.z() / vpPoint.w());
}

} // namespace PointyCloudPlugin
