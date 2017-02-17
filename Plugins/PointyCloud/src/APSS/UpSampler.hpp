#ifndef POINTYCLOUDPLUGIN_UPSAMPLER_H
#define POINTYCLOUDPLUGIN_UPSAMPLER_H

#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <PointyCloudSystem.hpp>

#include "PointyCloud.hpp"
namespace PointyCloudPlugin{

class UpSampler
{
public :

    UpSampler(float rayon);
    ~UpSampler();
    virtual void upSampleCloud(PointyCloud& cloud)=0;

protected :

    float m_rayon;
    std::vector<APoint> m_newpoints;
    PointyCloud* m_cloud;

    void upSamplePoint(const int& m, const int& indice);
    Ra::Core::Vector3 calculU(const Ra::Core::Vector3& normal);
    Ra::Core::Vector3 calculV(const Ra::Core::Vector3& normal,const Ra::Core::Vector3& u);

};
}
#endif // POINTYCLOUDPLUGIN_UPSAMPLER_H
