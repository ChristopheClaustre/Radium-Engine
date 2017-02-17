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

        void setInfluenceRadius(Scalar);
        void setBeta(Scalar);
        void setThreshold(int);
        void setM(int);
        void setUpsamplingMethod(UPSAMPLING_METHOD);
        void setProjectionMethod(PROJECTION_METHOD);
        void setOptimizationByOctree(bool);
        void setOptimizationByCUDA(bool);

    private:
        Ra::Core::Index m_meshIndex;
        std::string m_contentName;

        const Ra::Engine::Camera *m_camera;

        // the data
        std::shared_ptr<PointyCloud> m_originalCloud;
        std::shared_ptr<Ra::Engine::Mesh> m_workingCloud;

        // class for the APSS
        UpSampler* m_upsampler;
        UsefulPointsSelection* m_culling;
        std::shared_ptr<NeighborsSelection> m_selector;
        OrthogonalProjection* m_projection;
    };

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_POINTYCLOUDCOMPONENT_HPP
