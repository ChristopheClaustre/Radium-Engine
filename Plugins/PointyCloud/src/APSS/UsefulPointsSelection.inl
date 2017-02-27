#include "UsefulPointsSelection.hpp"

namespace PointyCloudPlugin
{

inline void UsefulPointsSelection::selectFromVisibility()
{
    Ra::Core::Matrix4 VP = m_camera->getProjMatrix() * m_camera->getViewMatrix();

    int indiceVisible = 0;
    for(int i = 0; i < m_cloud.size(); ++i)
    {
        Ra::Core::Vector4 point = Ra::Core::Vector4::Ones();
        point.head<3>() = m_cloud[i].pos();
        auto vpPoint = VP * point;

        auto X = vpPoint[0] / vpPoint[3];
        auto Y = vpPoint[1] / vpPoint[3];
        auto Z = vpPoint[2] / vpPoint[3];
        if (X <= 1 && Y <= 1 && Z <= 1 && X >= -1 && Y >= -1 && Z >= -1)
        {
            m_cloud.swap(indiceVisible,i);
            ++indiceVisible;
        }
    }

    m_N = indiceVisible;
}

inline void UsefulPointsSelection::selectFromOrientation()
{
    Ra::Core::Vector3 view = m_camera->getDirection();

    int indiceWellOriented = 0;
    if(m_camera->getProjType() == Ra::Engine::Camera::ProjType::ORTHOGRAPHIC)
    {
        for(int i = 0; i < m_cloud.size(); ++i)
        {
            if(view.dot(m_cloud[i].normal()) < 0) {
                m_cloud.swap(indiceWellOriented, i);
                ++indiceWellOriented;
            }
        }
    }
    else
    {
        for(int i = 0; i < m_cloud.size(); ++i)
        {
            Ra::Core::Vector3 vecteurCameraPoint(m_cloud[i].pos() - m_camera->getPosition());
            if(vecteurCameraPoint.dot(m_cloud[i].normal()) < 0) {
                m_cloud.swap(indiceWellOriented, i);
                ++indiceWellOriented;
            }
        }
    }

    m_N = indiceWellOriented;
}

} // namespace PointyCloudPlugin












