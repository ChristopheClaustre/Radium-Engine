#include "UpSamplerSimple.hpp"
namespace  PointyCloudPlugin {



UpSamplerSimple::UpSamplerSimple(float rayon,float M) : UpSampler(rayon), m_m(M)
{
}

UpSamplerSimple::~UpSamplerSimple()
{
}

void UpSamplerSimple::upSampleCloud(PointyCloud* cloud)
{
    m_cloud = cloud;
    m_newpoints.clear();
    const int &n = m_cloud->m_points.size() ;

    for ( uint i = 0 ; i < n ; i++ )
    {
        this->upSamplePoint(m_m, i);
    }
    m_cloud->m_points = m_newpoints;
}
}
