#include "UpSamplerSimple.hpp"

namespace  PointyCloudPlugin {

UpSamplerSimple::UpSamplerSimple(float rayon, float threshold,const Ra::Engine::Camera & camera) : UpSampler(rayon), m_threshold(threshold),m_camera(camera)
{
}

UpSamplerSimple::~UpSamplerSimple()
{
}

void UpSamplerSimple::upSampleCloud(PointyCloud& cloud)
{
    m_cloud = &cloud;
    m_newpoints.clear();
    const int &n = m_cloud->m_points.size() ;

    for ( uint i = 0 ; i < n ; i++ )
    {
        this->upSamplePoint(getM(i), i);
    }
    m_cloud->m_points = m_newpoints;
}

int UpSamplerSimple::getM(const int& indice)
{
    return round(calculEta(indice) * m_radius/ m_threshold);
}

int UpSamplerSimple::calculEta(const int& indice)
{
    float facteurObliquite;
    Ra::Core::Vector3 distPToCam = m_cloud->m_points[indice].pos() - m_camera.getPosition();
    float dist;
    //m_point are already normalized
    if(m_camera.getProjType() == Ra::Engine::Camera::ProjType::ORTHOGRAPHIC)
    {
        //        facteurObliquite = m_camera.getDirection().dot(m_cloud->m_points[indice].normal());
        //        float  d = m_camera.getDirection().dot(m_cloud->m_points[indice].pos());




    }
    else
    {
        //        facteurObliquite = distPToCam.normalized().dot(m_cloud->m_points[indice].normal());

        //    std::cerr << "FALSE" << std::endl;
    }
    const Ra::Core::Vector3 pointCam = pointInView(m_cloud->m_points[indice].pos());
    const Ra::Core::Vector3 &extremPoint1 = (pointCam + Ra::Core::Vector3( 1.0f* m_radius * 2 / m_camera.getWidth() , 0.0f , 0.0f) ) ;
    const Ra::Core::Vector3 &extremPoint2 = (pointCam + Ra::Core::Vector3(-1.0f* m_radius * 2 / m_camera.getWidth() , 0.0f , 0.0f) )  ;

    const Ra::Core::Vector2 extremPoint1Proj = project(extremPoint1);
    const Ra::Core::Vector2 extremPoint2Proj = project(extremPoint2);
//    dist = (extremPoint2Proj - extremPoint1Proj).norm();
//    std::cerr << "dist = " << ceil(dist) << std::endl;
    return ceil((extremPoint2Proj - extremPoint1Proj).norm());
}

} // namespace PointyCloudPlugin
