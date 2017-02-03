 
#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDRENDERER_HPP
#define POINTYCLOUDPLUGIN_POINTYCLOUDRENDERER_HPP

#include <Engine/RadiumEngine.hpp>
#include <Engine/Renderer/Renderer.hpp>

namespace Ra {
namespace Engine {
    class RenderData;
} // namespace Engine
} // namespace Ra

namespace PointyCloudPlugin
{
        class RA_ENGINE_API PointyCloudRenderer : public Ra::Engine::Renderer
        {
        public:
            PointyCloudRenderer( uint width, uint height, float splatSize );
            virtual ~PointyCloudRenderer();
            virtual std::string getRendererName() const override { return "PointyCloud Renderer"; }

            inline void setSplatSize(float size) {m_splatSize = size;}
            inline const float& getSplatSize() const {return m_splatSize;}

        protected:

            virtual void initializeInternal() override;
            virtual void resizeInternal() override;
            virtual void renderInternal( const Ra::Engine::RenderData& renderData ) override;

            virtual void updateStepInternal( const Ra::Engine::RenderData& renderData ) override {}
            virtual void debugInternal( const Ra::Engine::RenderData& renderData ) override {}
            virtual void uiInternal( const Ra::Engine::RenderData& renderData ) override {}
            virtual void postProcessInternal( const Ra::Engine::RenderData &renderData ) override {}

        private:
            void initShaders();
            void initBuffers();


        private:

            float m_splatSize;

            std::unique_ptr<Ra::Engine::FBO> m_fbo;
            std::unique_ptr<Ra::Engine::Texture> m_texture;

        };

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_POINTYCLOUDRENDERER_HPP
