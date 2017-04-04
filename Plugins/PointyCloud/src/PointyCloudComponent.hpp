#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDCOMPONENT_HPP
#define POINTYCLOUDPLUGIN_POINTYCLOUDCOMPONENT_HPP

#include <PointyCloudPlugin.hpp>

#include <Engine/Component/Component.hpp>

#include <APSS/Projection/OrthogonalProjection.hpp>
#include <APSS/UsefulPointsSelection.hpp>

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
    class PointyCloud;
    class NeighborsSelection;
    class UpSampler;

    class POINTY_PLUGIN_API PointyCloudComponent : public Ra::Engine::Component
    {
    public:
        PointyCloudComponent( const std::string& name, const Ra::Engine::Camera *camera);
        virtual ~PointyCloudComponent();

        virtual void initialize() override;

        void handlePointyCloudLoading(const Ra::Asset::GeometryData* data);

        /// Do APSS on the point cloud
        void computePointyCloud();

        void setInfluenceRadius(Scalar);
        void setSplatRadius(Scalar);
        void setThreshold(int);
        void setM(int);
        void setUpsamplingMethod(UPSAMPLING_METHOD);
        void setProjectionMethod(PROJECTION_METHOD);
        void setOptimizationByOctree(bool);
        void setOptimizationByCUDA(bool);

        void resetWorkingCloud();

    private:
        // set eligible flag on each points
        void setEligibleFlags();

    private:
        Ra::Core::Index m_meshIndex;
        std::string m_contentName;
        std::string m_cloudName;

        const Ra::Engine::Camera* m_camera;

        // the data
        std::shared_ptr<PointyCloud> m_originalCloud;
        std::shared_ptr<Ra::Engine::Mesh> m_workingCloud;

        // class for the APSS
        std::unique_ptr<UpSampler> m_upsampler;
        UsefulPointsSelection* m_culling;
        std::shared_ptr<NeighborsSelection> m_selector;
        OrthogonalProjection m_projection;

        // APSS attributes
        UPSAMPLING_METHOD m_upsamplingMethod;
        PROJECTION_METHOD m_projectionMethod;

        // APSS stats
        size_t m_count;
        float m_timeCulling;
        float m_timeUpsampling;
        float m_timeProjecting;
        float m_timeLoading;

    }; // class PointyCloudComponent

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_POINTYCLOUDCOMPONENT_HPP
