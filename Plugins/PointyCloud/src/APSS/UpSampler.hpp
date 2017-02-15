#ifndef UPSAMPLER_H
#define UPSAMPLER_H

#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <PointyCloudSystem.hpp>

namespace PointyCloudPlugin{


class UpSampler
{
protected :

    int getM(int indice);

public :

    UpSampler(float rayon) ;
    ~UpSampler();
    void upSampleCloud(std::shared_ptr<Ra::Engine::Mesh> cloud);

private :

    float m_rayon;
    Ra::Core::VectorArray<Ra::Core::Vector3> m_vectNormals;
    Ra::Core::VectorArray<Ra::Core::Vector3> m_vectVertices;
    std::shared_ptr<Ra::Engine::Mesh> m_cloud;

    void upSamplePoint(const int& m, const int& indice );
    Ra::Core::Vector3 calculU(const Ra::Core::Vector3& normal);
    Ra::Core::Vector3 calculV(const Ra::Core::Vector3& normal,const Ra::Core::Vector3& u);


};
}
#endif // UPSAMPLER_H
