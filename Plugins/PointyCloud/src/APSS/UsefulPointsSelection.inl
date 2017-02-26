#include "UsefulPointsSelection.hpp"

namespace PointyCloudPlugin
{

inline void UsefulPointsSelection::selectFromVisibility(PointyCloud& pc)
{
    Ra::Core::Matrix4 VP = m_camera->getProjMatrix() * m_camera->getViewMatrix();

    auto pointsFirst = pc.m_points.begin();
    auto pointsEnd = pc.m_points.end();
    for(auto pointsIt = pc.m_points.begin(); pointsIt!=pointsEnd; ++pointsIt)
    {
        Ra::Core::Vector4 point = Ra::Core::Vector4::Ones();
        point.head<3>() = pointsIt->pos();
        auto vpPoint = VP * point;

        auto X = vpPoint[0] / vpPoint[3];
        auto Y = vpPoint[1] / vpPoint[3];
        auto Z = vpPoint[2] / vpPoint[3];
        if (X <= 1 && Y <= 1 && Z <= 1 && X >= -1 && Y >= -1 && Z >= -1)
        {
            *pointsFirst++ = *pointsIt;
        }
    }

    pc.m_points.erase(pointsFirst, pointsEnd);
}

inline void UsefulPointsSelection::selectFromOrientation(PointyCloud& pc)
{
    Ra::Core::Vector3 view = m_camera->getDirection();

    auto pointsFirst = pc.m_points.begin();
    auto pointsEnd = pc.m_points.end();
    if(m_camera->getProjType() == Ra::Engine::Camera::ProjType::ORTHOGRAPHIC)
    {
        for(auto pointsIt = pc.m_points.begin(); pointsIt!=pointsEnd; ++pointsIt)
        {
            if(view.dot(pointsIt->normal()) < 0)
                *pointsFirst++ = *pointsIt;
        }
    }
    else
    {
        for(auto pointsIt = pc.m_points.begin(); pointsIt!=pointsEnd; ++pointsIt)
        {
            Ra::Core::Vector3 vecteurCameraPoint(pointsIt->pos() - m_camera->getPosition());
            if(vecteurCameraPoint.dot(pointsIt->normal()) < 0)
                *pointsFirst++ = *pointsIt;
        }
    }

    pc.m_points.erase(pointsFirst, pointsEnd);
}

} // namespace PointyCloudPlugin












