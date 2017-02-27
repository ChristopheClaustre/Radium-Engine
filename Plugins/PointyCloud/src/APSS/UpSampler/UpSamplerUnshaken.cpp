#include "UpSamplerUnshaken.hpp"

namespace PointyCloudPlugin {

UpSamplerUnshaken::UpSamplerUnshaken(int M) : UpSampler(), m_M(M)
{
}

UpSamplerUnshaken::~UpSamplerUnshaken()
{
}

void UpSamplerUnshaken::upSampleCloud(const PointyCloud &usefulPointCloud, int N)
{
    m_cloud.clear();
    #pragma omp parallel for num_threads(4)
    for ( uint i = 0 ; i < N ; i++ )
    {
        this->upSamplePoint(m_M, usefulPointCloud[i]);
    }
}

} // namespace PointyCloudPlugin
