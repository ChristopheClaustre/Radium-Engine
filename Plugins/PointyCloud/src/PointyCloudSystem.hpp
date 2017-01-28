#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDSYSTEM_HPP
#define POINTYCLOUDPLUGIN_POINTYCLOUDSYSTEM_HPP

#include "PointyCloudPlugin.hpp"

#include <Engine/System/System.hpp>

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
}

namespace PointyCloudPlugin
{
    class PC_PLUGIN_API PointyCloudSystem : public Ra::Engine::System
    {
    public:
        PointyCloudSystem();
        virtual ~PointyCloudSystem();

        virtual void handleAssetLoading( Ra::Engine::Entity* entity, const Ra::Asset::FileData* fileData ) override;

        virtual void generateTasks( Ra::Core::TaskQueue* taskQueue, const Ra::Engine::FrameInfo& frameInfo ) override;
    };

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_POINTYCLOUDSYSTEM_HPP
