 
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
            initShaders();
            initBuffers();
        }

        void PointyCloudRenderer::initShaders()
        {
            m_shaderMgr->addShaderProgram("Pointy", "../Shaders/Pointy/Pointy.vert.glsl", "../Shaders/Pointy/Pointy.frag.glsl");
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
                //TODO change glPolygonMode to handle oriented splat
                glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
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

                const Ra::Engine::ShaderProgram* shader = m_shaderMgr->getShaderProgram("Pointy");
                shader->bind();
                {
                    for(const auto& ro : m_pointyRenderObjects)
                        if( ro->isVisible() )
                        {
                            Ra::Core::Matrix4 M = ro->getTransformAsMatrix();
                            Ra::Core::Matrix4 N = M.inverse().transpose();

                            Ra::Engine::RenderParameters params;
                            if(m_lights.size()>0)
                                m_lights[0]->getRenderParameters( params );

                            params.bind(shader);

                            shader->setUniform( "transform.proj", renderData.projMatrix );
                            shader->setUniform( "transform.view", renderData.viewMatrix );
                            shader->setUniform( "transform.model", M );
                            shader->setUniform( "transform.worldNormal", N );
                            shader->setUniform( "splatSize", m_splatSize );

                            ro->getMesh()->render();
                        }
                }
                shader->unbind();

                // reset state
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
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
