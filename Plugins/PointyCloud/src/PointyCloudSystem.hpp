#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDSYSTEM_HPP
#define POINTYCLOUDPLUGIN_POINTYCLOUDSYSTEM_HPP

#include "PointyCloudPlugin.hpp"
#include "ComputePointyCloudTask.hpp"

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
        void setBeta(Scalar beta);
        void setThreshold(int threshold);
        void setM(int M);
        void setUpsamplingMethod(UPSAMPLING_METHOD upsampler);
        void setProjectionMethod(PROJECTION_METHOD projector);
        void setOptimizationByOctree(bool octree);
        void setOptimizationByCUDA(bool cuda);

        inline Scalar getSplatRadius() { return m_splatRadius; }
        inline Scalar getInfluenceRadius()  { return m_influenceRadius; }
        inline Scalar getBeta() { return m_beta; }
        inline int getThreshold() { return m_threshold; }
        inline int getM() { return m_M; }
        inline UPSAMPLING_METHOD getUpsamplingMethod() { return m_upsampler; }
        inline PROJECTION_METHOD getProjectionMethod() { return m_projector; }
        inline bool isOptimizedByOctree() { return m_octree; }
        inline bool isOptimizedByCUDA() { return m_cuda; }

    private:
        PointyCloudPlugin::PointyCloudRenderer * m_renderer;
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
    };

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_POINTYCLOUDSYSTEM_HPP
