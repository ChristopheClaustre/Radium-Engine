#ifndef POINTYCLOUDPLUGIN_UPSAMPLER_H
#define POINTYCLOUDPLUGIN_UPSAMPLER_H

#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <PointyCloudSystem.hpp>

#include <APSS/PointyCloud.hpp>

namespace PointyCloudPlugin
{

class UpSampler
{
public :
    UpSampler();
    virtual ~UpSampler();
    virtual void upSampleCloud(const PointyCloud& usefulPointCloud, int N)=0;
    inline PointyCloud& getUpsampledCloud() { return m_cloud; }

protected :
    PointyCloud m_cloud;

    void upSamplePoint(const int& m, const APoint& point);
    static Ra::Core::Vector3 calculU(const Ra::Core::Vector3& normal);
    static Ra::Core::Vector3 calculV(const Ra::Core::Vector3& normal, const Ra::Core::Vector3& u);

}; // class Upsampler

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_UPSAMPLER_H
