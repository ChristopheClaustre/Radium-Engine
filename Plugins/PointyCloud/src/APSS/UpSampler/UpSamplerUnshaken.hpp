#ifndef POINTYCLOUDPLUGIN_UPSAMPLERUNSHAKEN_H
#define POINTYCLOUDPLUGIN_UPSAMPLERUNSHAKEN_H

#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <PointyCloudSystem.hpp>

#include <APSS/PointyCloud.hpp>
#include <APSS/UpSampler/UpSampler.hpp>

namespace PointyCloudPlugin{

class UpSamplerUnshaken : public UpSampler
{
public :
    UpSamplerUnshaken(std::shared_ptr<PointyCloud> originalCloud, int UpsamplingInfo);
    virtual ~UpSamplerUnshaken();

    void setM(int M) { m_M = M; }

protected:
    void upSamplePointMaster(int index);

private :
    int m_M;
}; // class UpSamplerUnshaken

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_UPSAMPLERUNSHAKEN_H
