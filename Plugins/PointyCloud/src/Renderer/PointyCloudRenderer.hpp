 
#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDRENDERER_HPP
#define POINTYCLOUDPLUGIN_POINTYCLOUDRENDERER_HPP

#include <PointyCloudPlugin.hpp>

#include <Engine/RadiumEngine.hpp>
#include <Engine/Renderer/Renderer.hpp>

namespace Ra {
namespace Engine {
    class RenderData;
} // namespace Engine
} // namespace Ra

namespace PointyCloudPlugin
{
    class POINTY_PLUGIN_API PointyCloudRenderer : public Ra::Engine::Renderer
    {
    public:
        PointyCloudRenderer(uint width, uint height);
        virtual ~PointyCloudRenderer();
        virtual std::string getRendererName() const override { return "PointyCloud Renderer"; }

    protected:

        virtual void initializeInternal() override;
        virtual void resizeInternal() override;
        virtual void renderInternal( const Ra::Engine::RenderData& renderData ) override;

        virtual void updateStepInternal( const Ra::Engine::RenderData& renderData ) override {}
        virtual void debugInternal( const Ra::Engine::RenderData& renderData ) override {}
        virtual void uiInternal( const Ra::Engine::RenderData& renderData ) override {}
        virtual void postProcessInternal( const Ra::Engine::RenderData &renderData ) override {}

    private:
        void initBuffers();

    private:
        std::unique_ptr<Ra::Engine::FBO> m_fbo;
        std::unique_ptr<Ra::Engine::Texture> m_texture;

    }; // class PointyCloudRenderer

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_POINTYCLOUDRENDERER_HPP
