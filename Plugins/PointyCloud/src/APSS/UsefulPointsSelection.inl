#include "UsefulPointsSelection.hpp"

namespace PointyCloudPlugin
{

inline void UsefulPointsSelection::selectFromVisibility()
{
    Ra::Core::Matrix4 VP = m_camera->getProjMatrix() * m_camera->getViewMatrix();

    int indexVisible = 0;
    for(int i = 0; i < m_indices.size(); ++i)
    {
        Ra::Core::Vector4 point = Ra::Core::Vector4::Ones();
        point.head<3>() = m_originalCloud->at(m_indices[i]).pos();
        auto vpPoint = VP * point;

        auto X = vpPoint[0] / vpPoint[3];
        auto Y = vpPoint[1] / vpPoint[3];
        auto Z = vpPoint[2] / vpPoint[3];
        if (X <= 1 && Y <= 1 && Z <= 1 && X >= -1 && Y >= -1 && Z >= -1)
        {
            std::swap(m_indices[indexVisible], m_indices[i]);
            ++indexVisible;
        }
    }

    m_N = indexVisible;
}

inline void UsefulPointsSelection::selectFromOrientation()
{
    Ra::Core::Vector3 view = m_camera->getDirection();

    int indexWellOriented = 0;
    if(m_camera->getProjType() == Ra::Engine::Camera::ProjType::ORTHOGRAPHIC)
    {
        for(int i = 0; i < m_indices.size(); ++i)
        {
            if(view.dot(m_originalCloud->at(m_indices[i]).normal()) < 0) {
                std::swap(m_indices[indexWellOriented], m_indices[i]);
                ++indexWellOriented;
            }
        }
    }
    else
    {
        for(int i = 0; i < m_indices.size(); ++i)
        {
            Ra::Core::Vector3 vecteurCameraPoint(m_originalCloud->at(m_indices[i]).pos() - m_camera->getPosition());
            if(vecteurCameraPoint.dot(m_originalCloud->at(m_indices[i]).normal()) < 0) {
                std::swap(m_indices[indexWellOriented], m_indices[i]);
                ++indexWellOriented;
            }
        }
    }

    m_N = indexWellOriented;
}

} // namespace PointyCloudPlugin












