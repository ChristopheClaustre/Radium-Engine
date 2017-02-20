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

    //m_point are already normalized
    if(m_camera.getProjType() == Ra::Engine::Camera::ProjType::ORTHOGRAPHIC)
    {
        facteurObliquite = m_camera.getDirection().normalized().dot(m_cloud->m_points[indice].normal());
    }
    else
    {
        facteurObliquite = distPToCam.normalized().dot(m_cloud->m_points[indice].normal());
    }
    facteurObliquite = (facteurObliquite+1.0f)*5;

    std::cerr <<  facteurObliquite   << std::endl;
    return (10 / log2( distPToCam.norm()  + 2  ))*facteurObliquite ;
}
}
