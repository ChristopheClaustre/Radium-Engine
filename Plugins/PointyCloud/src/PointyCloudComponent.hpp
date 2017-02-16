#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDCOMPONENT_HPP
#define POINTYCLOUDPLUGIN_POINTYCLOUDCOMPONENT_HPP

#include <PointyCloudPlugin.hpp>

#include <Core/Mesh/MeshTypes.hpp>
#include <Core/Mesh/TriangleMesh.hpp>
#include <Core/Containers/MakeShared.hpp>

#include <Engine/Component/Component.hpp>
#include <Engine/Renderer/Camera/Camera.hpp>

#include <APSS/PointyCloud.hpp>

namespace Ra
{
    namespace Engine
    {
        struct RenderTechnique;
        class Mesh;
        class Camera;
    }

    namespace Asset
    {
        class GeometryData;
    }
}

namespace PointyCloudPlugin
{
    class UsefulPointsSelection;
    class OrthogonalProjection;
    class NeighborsSelection;
    class UpSampler;

    class POINTY_PLUGIN_API PointyCloudComponent : public Ra::Engine::Component
    {
    public:
        PointyCloudComponent( const std::string& name, const Ra::Engine::Camera *camera);
        virtual ~PointyCloudComponent();


        virtual void initialize() override;

        void handlePointyCloudLoading(const Ra::Asset::GeometryData* data);

        /// Returns the index of the associated RO (the display mesh)
        Ra::Core::Index getRenderObjectIndex() const;

        /// Do APSS on the point cloud
        void computePointyCloud();

        void setInfluenceRadius(float);
        void setBeta(float);
        void setThreshold(int);
        void setM(int);
        void setUpsamplingMethod(UPSAMPLING_METHOD);
        void setProjectionMethod(PROJECTION_METHOD);
        void setOptimizationByOctree(bool);
        void setOptimizationByCUDA(bool);

    private:
        Ra::Core::Index m_meshIndex;
        std::string m_contentName;

        std::shared_ptr<PointyCloud> m_originalCloud;
        std::shared_ptr<Ra::Engine::Mesh> m_workingCloud;

        const Ra::Engine::Camera *m_camera;

        UsefulPointsSelection* m_culling;
        UpSampler* m_upsampler;
        OrthogonalProjection* m_projection;
        NeighborsSelection* m_selector;

    };

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_POINTYCLOUDCOMPONENT_HPP
