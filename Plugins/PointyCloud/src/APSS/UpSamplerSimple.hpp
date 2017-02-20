#ifndef UPSAMPLERSIMPLE_H
#define UPSAMPLERSIMPLE_H

#include "UpSampler.hpp"
#include <Engine/Renderer/Camera/Camera.hpp>

namespace PointyCloudPlugin{
class UpSamplerSimple:public UpSampler
{
public:
    UpSamplerSimple(float rayon, float threshold, const Ra::Engine::Camera& camera);
    ~UpSamplerSimple();
    virtual void upSampleCloud(PointyCloud& cloud);
protected:
    float m_threshold;
    const Ra::Engine::Camera &m_camera;

    int getM(const int& indice);
    int calculEta(const int& indice);
};

}

#endif // UPSAMPLERSIMPLE_H