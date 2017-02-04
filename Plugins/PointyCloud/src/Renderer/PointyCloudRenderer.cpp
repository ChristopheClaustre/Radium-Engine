 
#include "src/Renderer/PointyCloudRenderer.hpp"

#include <Engine/RadiumEngine.hpp>

#include <Engine/Renderer/OpenGL/OpenGL.hpp>
#include <Engine/Renderer/OpenGL/FBO.hpp>
#include <Engine/Renderer/RenderTechnique/ShaderProgramManager.hpp>
#include <Engine/Renderer/RenderTechnique/ShaderProgram.hpp>
#include <Engine/Renderer/RenderTechnique/RenderParameters.hpp>
#include <Engine/Renderer/Texture/Texture.hpp>
#include <Engine/Renderer/Light/Light.hpp>

namespace PointyCloudPlugin
{
        namespace
        {
            const GLenum buffers[] =
            {
                GL_COLOR_ATTACHMENT0,
                GL_COLOR_ATTACHMENT1,
                GL_COLOR_ATTACHMENT2,
                GL_COLOR_ATTACHMENT3,
                GL_COLOR_ATTACHMENT4,
                GL_COLOR_ATTACHMENT5,
                GL_COLOR_ATTACHMENT6,
                GL_COLOR_ATTACHMENT7
            };
        }

        PointyCloudRenderer::PointyCloudRenderer(uint width, uint height , float splatSize) :
            Ra::Engine::Renderer(width, height),
            m_splatSize(splatSize)
        {

        }

        PointyCloudRenderer::~PointyCloudRenderer()
        {
            Ra::Engine::ShaderProgramManager::destroyInstance();
        }

        void PointyCloudRenderer::initializeInternal()
        {
            initBuffers();
        }

        void PointyCloudRenderer::initBuffers()
        {
            m_fbo.reset(new Ra::Engine::FBO(Ra::Engine::FBO::Component_Color | Ra::Engine::FBO::Component_Depth, m_width, m_height));

            m_texture.reset(new Ra::Engine::Texture("Pointy Depth"));
            m_texture->internalFormat = GL_DEPTH_COMPONENT24;
            m_texture->dataType       = GL_UNSIGNED_INT;

            m_secondaryTextures["Pointy Depth"] = m_texture.get();
        }

        void PointyCloudRenderer::renderInternal( const Ra::Engine::RenderData& renderData )
        {
            m_fbo->useAsTarget(m_width, m_height);
            {
                glPointSize(m_splatSize);

                GL_ASSERT( glDepthMask( GL_TRUE ) );
                GL_ASSERT( glColorMask( 1, 1, 1, 1 ) );
                GL_ASSERT( glDrawBuffers( 1, buffers ) );

                const Ra::Core::Colorf gray = Ra::Core::Colors::FromChars<Ra::Core::Colorf>(42, 42, 42, 0);
                const float one = 1.0;

                GL_ASSERT( glClearBufferfv( GL_DEPTH, 0, &one ) );
                GL_ASSERT( glClearBufferfv( GL_COLOR, 0, gray.data() ) );

                GL_ASSERT( glEnable( GL_DEPTH_TEST ) );
                GL_ASSERT( glDepthFunc( GL_LESS ) );
                GL_ASSERT( glDepthMask( GL_TRUE ) );
                GL_ASSERT( glDisable( GL_BLEND ) );

                GL_ASSERT( glDrawBuffers( 1, buffers ) );

                for(const auto& ro : m_pointyRenderObjects)
                    if( ro->isVisible() )
                    {
                        Ra::Engine::RenderParameters params;
                        if(m_lights.size()>0)
                            m_lights[0]->getRenderParameters( params );

                        //TODO: peut être changer cette ajout de paramètre
                        // normalement c'est les ligthparams uniquement...
                        // mais bon j'ai l'impression qu'il n'existe aucun autre moyen...
                        params.addParameter("splatSize", m_splatSize);
                        ro->render(params, renderData);
                    }
            }
            m_fbo->unbind();
        }

        void PointyCloudRenderer::resizeInternal()
        {
            m_texture->Generate(m_width, m_height, GL_DEPTH_COMPONENT);

            m_fbo->bind();
                m_fbo->setSize(m_width, m_height);
                m_fbo->attachTexture(GL_DEPTH_ATTACHMENT , m_texture.get());
                m_fbo->attachTexture(GL_COLOR_ATTACHMENT0, m_fancyTexture.get());
                m_fbo->check();
            m_fbo->unbind(true);

            GL_CHECK_ERROR;

            // Reset framebuffer state
            GL_ASSERT( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

            GL_ASSERT( glDrawBuffer( GL_BACK ) );
            GL_ASSERT( glReadBuffer( GL_BACK ) );
        }

} // namespace PointyCloudPlugin
