#ifndef POINTYCLOUDPLUGIN_UPSAMPLERUNSHAKEN_H
#define POINTYCLOUDPLUGIN_UPSAMPLERUNSHAKEN_H

#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <PointyCloudSystem.hpp>

#include "PointyCloud.hpp"
#include "UpSampler.hpp"
namespace PointyCloudPlugin{

class UpSamplerUnshaken:public UpSampler
{

public :

    UpSamplerUnshaken(float rayon,float M) ;
    ~UpSamplerUnshaken();
    virtual void upSampleCloud(PointyCloud* cloud);

private :
    int m_m;
};
}
#endif // POINTYCLOUDPLUGIN_UPSAMPLERUNSHAKEN_H
