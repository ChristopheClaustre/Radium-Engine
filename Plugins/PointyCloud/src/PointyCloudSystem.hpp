#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDSYSTEM_HPP
#define POINTYCLOUDPLUGIN_POINTYCLOUDSYSTEM_HPP

#include "PointyCloudPlugin.hpp"

#include <Engine/System/System.hpp>
#include <GuiBase/Viewer/Viewer.hpp>
#include <Renderer/PointyCloudRenderer.hpp>

namespace Ra
{
    namespace Core
    {
        struct TriangleMesh;
    }
}

namespace Ra
{
    namespace Engine
    {
        class Entity;
        struct RenderTechnique;
        class Component;
//        class PointyCloudRenderer;
    }
}

//namespace Ra
//{
//    namespace GUI
//    {
//        class Viewer;
//    }
//}

namespace PointyCloudPlugin
{
    class PointyCloudComponent;
    enum UPSAMPLING_METHOD;
    enum PROJECTION_METHOD;
}

namespace PointyCloudPlugin
{
    class POINTY_PLUGIN_API PointyCloudSystem : public Ra::Engine::System
    {
    public:
        PointyCloudSystem(Ra::Gui::Viewer * viewer);
        virtual ~PointyCloudSystem();

        virtual void handleAssetLoading( Ra::Engine::Entity* entity, const Ra::Asset::FileData* fileData ) override;

        virtual void generateTasks( Ra::Core::TaskQueue* taskQueue, const Ra::Engine::FrameInfo& frameInfo ) override;

        std::vector<PointyCloudComponent*> getComponents();

        void setSplatRadius(float splatRadius);
        void setInfluenceRadius(float influenceRadius);
        void setBeta(float beta);
        void setThreshold(float threshold);
        void setUpsamplingMethod(UPSAMPLING_METHOD upsampler);
        void setProjectionMethod(PROJECTION_METHOD projector);
        void setOptimizationByOctree(bool octree);
        void setOptimizationByCUDA(bool cuda);

        float getSplatRadius() { return m_splatRadius; }
        float getInfluenceRadius()  { return m_influenceRadius; }
        float getBeta() { return m_beta; }
        float getThreshold() { return m_threshold; }
        UPSAMPLING_METHOD getUpsamplingMethod() { return m_upsampler; }
        PROJECTION_METHOD getProjectionMethod() { return m_projector; }
        bool isOptimizedByOctree() { return m_octree; }
        bool isOptimizedByCUDA() { return m_cuda; }

    private:
        PointyCloudPlugin::PointyCloudRenderer * m_renderer;
        int m_rendererIndex;
        Ra::Gui::Viewer * m_viewer;
        std::vector<PointyCloudComponent*> pointyCloudComponentList;

        float m_splatRadius;
        float m_influenceRadius;
        float m_beta;
        float m_threshold;
        UPSAMPLING_METHOD m_upsampler;
        PROJECTION_METHOD m_projector;
        bool m_octree;
        bool m_cuda;
    };

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_POINTYCLOUDSYSTEM_HPP
