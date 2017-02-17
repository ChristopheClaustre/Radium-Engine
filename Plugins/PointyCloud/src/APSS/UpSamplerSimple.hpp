#ifndef UPSAMPLERSIMPLE_H
#define UPSAMPLERSIMPLE_H

#include "UpSampler.hpp"

namespace PointyCloudPlugin{
class UpSamplerSimple:public UpSampler
{
public:
    UpSamplerSimple(float rayon, float M);
    ~UpSamplerSimple();
    virtual void upSampleCloud(PointyCloud* cloud);

private :
    int m_m;
};

}

#endif // UPSAMPLERSIMPLE_H
