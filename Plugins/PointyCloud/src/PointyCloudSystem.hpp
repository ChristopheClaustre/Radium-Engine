#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDSYSTEM_HPP
#define POINTYCLOUDPLUGIN_POINTYCLOUDSYSTEM_HPP

#include <PointyCloudPlugin.hpp>

#include <Engine/System/System.hpp>

namespace Ra
{
    namespace Core
    {
        struct TriangleMesh;
    }

    namespace GuiBase {
        class Viewer;
    } // namespace Viewer

    namespace Engine
    {
        class Entity;
        struct RenderTechnique;
        class Component;
    }
}

namespace PointyCloudPlugin
{
    class PointyCloudComponent;
    enum UPSAMPLING_METHOD;
    enum PROJECTION_METHOD;
}

namespace PointyCloudPlugin
{
    class PointyCloudRenderer;

    class POINTY_PLUGIN_API PointyCloudSystem : public Ra::Engine::System
    {
    public:
        PointyCloudSystem(Ra::Gui::Viewer * viewer);
        virtual ~PointyCloudSystem();

        virtual void handleAssetLoading( Ra::Engine::Entity* entity, const Ra::Asset::FileData* fileData ) override;

        virtual void generateTasks( Ra::Core::TaskQueue* taskQueue, const Ra::Engine::FrameInfo& frameInfo ) override;

        std::vector<PointyCloudComponent*> getComponents();

        void setSplatRadius(Scalar splatRadius);
        void setInfluenceRadius(Scalar influenceRadius);
        void setThreshold(int threshold);
        void setM(int M);
        void setUpsamplingMethod(UPSAMPLING_METHOD upsampler);
        void setProjectionMethod(PROJECTION_METHOD projector);
        void setOptimizationByOctree(bool octree);
        void setOptimizationByCUDA(bool cuda);
        void setAPSS(bool apss);
        void setRenderer(bool renderer);

        inline const Scalar& getSplatRadius() const { return m_splatRadius; }
        inline const Scalar& getInfluenceRadius()  const { return m_influenceRadius; }
        inline const Scalar& getBeta() const { return m_beta; }
        inline const int& getThreshold() const { return m_threshold; }
        inline const int& getM() const { return m_M; }
        inline const UPSAMPLING_METHOD& getUpsamplingMethod() const { return m_upsampler; }
        inline const PROJECTION_METHOD& getProjectionMethod() const { return m_projector; }
        inline const bool& isOptimizedByOctree() const { return m_octree; }
        inline const bool& isOptimizedByCUDA() const { return m_cuda; }
        inline const bool& isAPSSused() const { return m_APSS; }
        inline const bool& isRendererUsed() const { return m_rendererUsed; }

    private:
        PointyCloudRenderer * m_renderer;
        int m_rendererIndex;
        Ra::Gui::Viewer * m_viewer;
        std::vector<PointyCloudComponent*> pointyCloudComponentList;

        Scalar m_splatRadius;
        Scalar m_influenceRadius;
        Scalar m_beta;
        int m_threshold;
        int m_M;
        UPSAMPLING_METHOD m_upsampler;
        PROJECTION_METHOD m_projector;
        bool m_octree;
        bool m_cuda;
        bool m_APSS;
        bool m_rendererUsed;
    };

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_POINTYCLOUDSYSTEM_HPP
