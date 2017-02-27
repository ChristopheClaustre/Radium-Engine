#ifndef POINTYCLOUDPLUGIN_UPSAMPLER_H
#define POINTYCLOUDPLUGIN_UPSAMPLER_H

#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <PointyCloudSystem.hpp>

#include <APSS/PointyCloud.hpp>

namespace PointyCloudPlugin
{

typedef struct {
    int m_M;
    int m_begin;
} UpsamplingInfo;

class UpSampler
{
public :
    UpSampler(std::shared_ptr<PointyCloud> originalCloud);
    virtual ~UpSampler();
    void upSampleCloud(const std::vector<unsigned int>& indices, int m_count);
    inline PointyCloud& getUpsampledCloud() { return *m_prevCloud; }

    inline void resetUpsamplingInfo() { m_prevUpsamplingInfo->clear(); m_upsamplingInfo->clear(); }

protected:
    void upSamplePoint(const int& m, const APoint& point, int index);
    virtual void upSamplePointMaster(int indice)=0;
    static Ra::Core::Vector3 calculU(const Ra::Core::Vector3& normal);
    static Ra::Core::Vector3 calculV(const Ra::Core::Vector3& normal, const Ra::Core::Vector3& u);

protected :
    std::shared_ptr<PointyCloud> m_originalCloud;
    PointyCloud* m_cloud;
    PointyCloud* m_prevCloud;
    std::map<unsigned int, UpsamplingInfo> * m_upsamplingInfo;
    std::map<unsigned int, UpsamplingInfo> * m_prevUpsamplingInfo;

    int m_count;
}; // class Upsampler

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_UPSAMPLER_H
