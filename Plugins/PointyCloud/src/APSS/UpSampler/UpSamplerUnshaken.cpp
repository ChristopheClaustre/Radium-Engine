#include "UpSamplerUnshaken.hpp"

namespace PointyCloudPlugin {

UpSamplerUnshaken::UpSamplerUnshaken(int M) : UpSampler(), m_M(M)
{
}

UpSamplerUnshaken::~UpSamplerUnshaken()
{
}

void UpSamplerUnshaken::upSampleCloud(PointyCloud& cloud)
{
    m_cloud = &cloud;
    m_newpoints.clear();
    const int &n = m_cloud->m_points.size() ;

    for ( uint i = 0 ; i < n ; i++ )
    {
        this->upSamplePoint(m_M, i);
    }
    m_cloud->m_points = m_newpoints;
}

} // namespace PointyCloudPlugin
