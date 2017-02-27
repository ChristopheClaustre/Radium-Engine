#include "UpSamplerUnshaken.hpp"

namespace PointyCloudPlugin {

UpSamplerUnshaken::UpSamplerUnshaken(std::shared_ptr<PointyCloud> originalCloud, int M)
    : UpSampler(originalCloud), m_M(M)
{
}

UpSamplerUnshaken::~UpSamplerUnshaken()
{
}

void UpSamplerUnshaken::upSamplePointMaster(int index)
{
    this->upSamplePoint(m_M, m_originalCloud->at(index), index);
    std::cout << "master :" << m_count << std::endl << std::flush;
}

} // namespace PointyCloudPlugin
