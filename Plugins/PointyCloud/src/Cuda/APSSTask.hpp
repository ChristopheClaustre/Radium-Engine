#ifndef POINTYCLOUDPLUGIN_APSSTASK_HPP
#define POINTYCLOUDPLUGIN_APSSTASK_HPP

#include <Core/Tasks/Task.hpp>

#include <Cuda/APSS.h>

#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <Core/Math/LinearAlgebra.hpp>
#include <Engine/Renderer/Camera/Camera.hpp>

namespace PointyCloudPlugin {

    class APSSTask : public Ra::Core::Task
    {
    public:
        APSSTask(Cuda::APSS* apss, std::shared_ptr<Ra::Engine::Mesh> mesh, const Ra::Engine::Camera* camera) :
            m_apss(apss), m_mesh(mesh), m_camera(camera) {}
        ~APSSTask() {}

        virtual std::string getName() const override {return "APSS";}

        virtual void process() override
        {
            // APSS steps
            m_apss->select(m_camera->getPosition(), m_camera->getDirection());
            m_apss->upsample(/*APSS parameters*/);
            m_apss->project(/*APSS parameters*/);
            m_apss->finalize();

            // get results
            size_t size = m_apss->sizeFinal();
            const Ra::Core::Vector3* positions = m_apss->positionFinal();
            const Ra::Core::Vector3* normals = m_apss->normalFinal();
            const Ra::Core::Vector4* colors = m_apss->colorFinal();
            const Scalar* splatSizes = m_apss->splatSizeFinal();

            // send results to target mesh
            m_mesh->loadPointyCloud(size, positions, normals, colors, splatSizes);
        }

    private:

        Cuda::APSS* m_apss;
        std::shared_ptr<Ra::Engine::Mesh> m_mesh;

        const Ra::Engine::Camera* m_camera;
    };

} // namespace PointyCloudPlugin


#endif // POINTYCLOUDPLUGIN_APSSTASK_HPP
