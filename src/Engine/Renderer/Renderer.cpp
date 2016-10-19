#include <Engine/Renderer/Renderer.hpp>

#include <iostream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <Core/Log/Log.hpp>
#include <Core/Math/ColorPresets.hpp>
#include <Core/Mesh/MeshUtils.hpp>
#include <Core/Mesh/MeshPrimitives.hpp>

#include <Engine/RadiumEngine.hpp>
#include <Engine/Managers/AssetManager.hpp>
#include <Engine/Renderer/OpenGL/OpenGL.hpp>
#include <Engine/Renderer/OpenGL/FBO.hpp>
#include <Engine/Renderer/RenderTechnique/ShaderProgram.hpp>
#include <Engine/Renderer/RenderTechnique/RenderParameters.hpp>
#include <Engine/Renderer/RenderTechnique/RenderTechnique.hpp>
#include <Engine/Renderer/RenderTechnique/Material.hpp>
#include <Engine/Renderer/Light/Light.hpp>
#include <Engine/Renderer/Light/DirLight.hpp>
#include <Engine/Renderer/Light/DirLight.hpp>
#include <Engine/Renderer/Light/PointLight.hpp>
#include <Engine/Renderer/Light/SpotLight.hpp>
#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <Engine/Renderer/Texture/TextureManager.hpp>
#include <Engine/Renderer/Texture/Texture.hpp>

namespace Ra
{
    namespace Engine
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

        Renderer::Renderer( uint width, uint height )
            : m_width( width )
            , m_height( height )
            , m_renderQueuesUpToDate( false )
            , m_drawDebug( true )
            , m_wireframe(false)
            , m_postProcessEnabled(true)
        {
        }

        Renderer::~Renderer()
        {
            TextureManager::destroyInstance();
        }

        void Renderer::initialize()
        {
            // Initialize managers
            m_assetMgr = AssetManager::getInstance();
            m_roMgr = RadiumEngine::getInstance()->getRenderObjectManager();
            TextureManager::createInstance();

            m_drawScreenShader = m_assetMgr->shaderProgram(m_assetMgr->createShaderProgram("../Shaders/Basic2D.vert.glsl", "../Shaders/DrawScreen.frag.glsl"));
            m_pickingShader    = m_assetMgr->shaderProgram(m_assetMgr->createShaderProgram("../Shaders/Picking.vert.glsl", "../Shaders/Picking.frag.glsl"));

            m_depthTexture = m_assetMgr->texture(m_assetMgr->createTexture("Depth"));
            m_depthTexture->internalFormat = GL_DEPTH_COMPONENT24;
            m_depthTexture->dataType = GL_UNSIGNED_INT;

            // Picking
            m_pickingFbo.reset(new FBO(FBO::Component( FBO::Component_Color | FBO::Component_Depth ), m_width, m_height));
            m_pickingTexture = m_assetMgr->texture(m_assetMgr->createTexture("Picking"));
            m_pickingTexture->internalFormat = GL_RGBA32I;
            m_pickingTexture->dataType = GL_INT;
            m_pickingTexture->minFilter = GL_NEAREST;
            m_pickingTexture->magFilter = GL_NEAREST;

            // Final texture
            m_fancyTexture = m_assetMgr->texture(m_assetMgr->createTexture("Final"));
            m_fancyTexture->internalFormat = GL_RGBA32F;
            m_fancyTexture->dataType = GL_FLOAT;

            m_displayedTexture = m_fancyTexture;
            m_secondaryTextures["Picking Texture"] = m_pickingTexture;

            // Quad mesh
            Core::TriangleMesh mesh = Core::MeshUtils::makeZNormalQuad(Core::Vector2( -1.f, 1.f));

            m_quadMesh = m_assetMgr->mesh(m_assetMgr->createMesh("quad"));
            m_quadMesh->loadGeometry( mesh );
            m_quadMesh->updateGL();

            initializeInternal();

            resize( m_width, m_height );
        }

        void Renderer::render( const RenderData& data )
        {
            CORE_ASSERT( RadiumEngine::getInstance() != nullptr, "Engine is not initialized." );

            std::lock_guard<std::mutex> renderLock( m_renderMutex );
            CORE_UNUSED( renderLock );

            m_timerData.renderStart = Core::Timer::Clock::now();

            // 0. Save eventual already bound FBO (e.g. QtOpenGLWidget)
            saveExternalFBOInternal();

            // 1. Gather render objects if needed
            feedRenderQueuesInternal( data );

            m_timerData.feedRenderQueuesEnd = Core::Timer::Clock::now();

            // 2. Update them (from an opengl point of view)
            // FIXME(Charly): Maybe we could just update objects if they need it
            // before drawing them, that would be cleaner (performance problem ?)
            updateRenderObjectsInternal( data );
            m_timerData.updateEnd = Core::Timer::Clock::now();

            // 3. Do picking if needed
            m_pickingResults.clear();
            if ( !m_pickingQueries.empty() )
            {
                doPicking( data );
            }
            m_lastFramePickingQueries = m_pickingQueries;
            m_pickingQueries.clear();

            updateStepInternal( data );

            // 4. Do the rendering.
            renderInternal( data );
            m_timerData.mainRenderEnd = Core::Timer::Clock::now();

            // 5. Post processing
            postProcessInternal( data );
            m_timerData.postProcessEnd = Core::Timer::Clock::now();

            // 6. Debug
            debugInternal( data );

            // 7. Draw UI
            uiInternal( data );

            // 8. Write image to framebuffer.
            drawScreenInternal();
            m_timerData.renderEnd = Core::Timer::Clock::now();

            // 9. Tell renderobjects they have been drawn (to decreaase the counter)
            notifyRenderObjectsRenderingInternal();
        }

        void Renderer::saveExternalFBOInternal()
        {
            GL_ASSERT( glGetIntegerv( GL_FRAMEBUFFER_BINDING, &m_qtPlz ) );
        }

        void Renderer::updateRenderObjectsInternal( const RenderData& renderData )
        {
            for ( auto& ro : m_fancyRenderObjects ) ro->updateGL();
            for ( auto& ro : m_xrayRenderObjects  ) ro->updateGL();
            for ( auto& ro : m_debugRenderObjects ) ro->updateGL();
            for ( auto& ro : m_uiRenderObjects    ) ro->updateGL();
        }

        void Renderer::feedRenderQueuesInternal( const RenderData& renderData )
        {
            m_fancyRenderObjects.clear();
            m_debugRenderObjects.clear();
            m_uiRenderObjects.clear();
            m_xrayRenderObjects.clear();

            m_roMgr->getRenderObjectsByType( renderData, m_fancyRenderObjects, RenderObjectType::Fancy );
            m_roMgr->getRenderObjectsByType( renderData, m_debugRenderObjects, RenderObjectType::Debug );
            m_roMgr->getRenderObjectsByType( renderData, m_uiRenderObjects,    RenderObjectType::UI );

            for ( auto it = m_fancyRenderObjects.begin(); it != m_fancyRenderObjects.end(); )
            {
                if ( (*it)->isXRay() )
                {
                    m_xrayRenderObjects.push_back( *it );
                    it = m_fancyRenderObjects.erase( it );
                }
                else
                {
                    ++it;
                }
            }

            for ( auto it = m_debugRenderObjects.begin(); it != m_debugRenderObjects.end(); )
            {
                if ( (*it)->isXRay() )
                {
                    m_xrayRenderObjects.push_back( *it );
                    it = m_debugRenderObjects.erase( it );
                }
                else
                {
                    ++it;
                }
            }

            for ( auto it = m_uiRenderObjects.begin(); it != m_uiRenderObjects.end(); )
            {
                if ( (*it)->isXRay() )
                {
                    m_xrayRenderObjects.push_back( *it );
                    it = m_uiRenderObjects.erase( it );
                }
                else
                {
                    ++it;
                }
            }
        }

        void Renderer::doPicking( const RenderData& renderData )
        {
            m_pickingResults.reserve( m_pickingQueries.size() );

            m_pickingFbo->useAsTarget();

            GL_ASSERT( glDepthMask( GL_TRUE ) );
            GL_ASSERT( glColorMask( 1, 1, 1, 1 ) );
            GL_ASSERT( glDrawBuffers( 1, buffers ) );

            float clearDepth = 1.0;
            int clearColor[] = { -1, -1, -1, -1 };

            GL_ASSERT(glClearBufferiv(GL_COLOR, 0, clearColor));
            GL_ASSERT(glClearBufferfv(GL_DEPTH, 0, &clearDepth));

            const ShaderProgram* shader = m_pickingShader;
            shader->bind();

            GL_ASSERT( glEnable( GL_DEPTH_TEST ) );
            GL_ASSERT( glDepthFunc( GL_LESS ) );

            for ( const auto& ro : m_fancyRenderObjects )
            {
                if ( ro->isVisible() )
                {
                    int id = ro->idx.getValue();
                    shader->setUniform( "objectId", id );

                    Core::Matrix4 M = ro->getTransformAsMatrix();
                    shader->setUniform( "transform.proj", renderData.projMatrix );
                    shader->setUniform( "transform.view", renderData.viewMatrix );
                    shader->setUniform( "transform.model", M );

                    ro->getRenderTechnique()->material->bind( shader );

                    // render
                    ro->getMesh()->render();
                }
            }

            // Draw debug objects
            GL_ASSERT( glClear( GL_DEPTH_BUFFER_BIT ) );
            if ( m_drawDebug )
            {
                for ( const auto& ro : m_debugRenderObjects )
                {
                    if ( ro->isVisible() )
                    {
                        int id = ro->idx.getValue();
                        shader->setUniform( "objectId", id );

                        Core::Matrix4 M = ro->getTransformAsMatrix();
                        shader->setUniform( "transform.proj", renderData.projMatrix );
                        shader->setUniform( "transform.view", renderData.viewMatrix );
                        shader->setUniform( "transform.model", M );

                        ro->getRenderTechnique()->material->bind( shader );

                        // render
                        ro->getMesh()->render();
                    }
                }
            }

            // Draw xrayed objects on top of normal objects
            GL_ASSERT( glClear( GL_DEPTH_BUFFER_BIT ) );
            if ( m_drawDebug )
            {
                for ( const auto& ro : m_xrayRenderObjects )
                {
                    if ( ro->isVisible() )
                    {
                        int id = ro->idx.getValue();
                        shader->setUniform( "objectId", id );

                        Core::Matrix4 M = ro->getTransformAsMatrix();
                        shader->setUniform( "transform.proj", renderData.projMatrix );
                        shader->setUniform( "transform.view", renderData.viewMatrix );
                        shader->setUniform( "transform.model", M );

                        ro->getRenderTechnique()->material->bind( shader );

                        // render
                        ro->getMesh()->render();
                    }
                }
            }


            // Always draw ui stuff on top of everything
            GL_ASSERT( glClear( GL_DEPTH_BUFFER_BIT ) );
            for ( const auto& ro : m_uiRenderObjects )
            {
                if ( ro->isVisible() )
                {
                    int id = ro->idx.getValue();
                    shader->setUniform( "objectId", id );

                    Core::Matrix4 M = ro->getTransformAsMatrix();
                    Core::Matrix4 MV = renderData.viewMatrix * M;
                    Scalar d = MV.block<3, 1>( 0, 3 ).norm();

                    Core::Matrix4 S = Core::Matrix4::Identity();
                    S( 0, 0 ) = S( 1, 1 ) = S( 2, 2 ) = d;

                    M = M * S;

                    shader->setUniform( "transform.proj", renderData.projMatrix );
                    shader->setUniform( "transform.view", renderData.viewMatrix );
                    shader->setUniform( "transform.model", M );

                    ro->getRenderTechnique()->material->bind( shader );

                    // render
                    ro->getMesh()->render();
                }
            }

            GL_ASSERT( glReadBuffer( GL_COLOR_ATTACHMENT0 ) );

            for ( const auto& query : m_pickingQueries )
            {
                int picking_result[4];
                GL_ASSERT( glReadPixels( query.m_screenCoords.x(), query.m_screenCoords.y(),
                                         1, 1, GL_RGBA_INTEGER, GL_INT, picking_result ) );

                m_pickingResults.push_back( picking_result[0] );
            }

            m_pickingFbo->unbind();
        }

        void Renderer::drawScreenInternal()
        {
            if ( m_qtPlz == 0 )
            {
                GL_ASSERT( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );
                glDrawBuffer( GL_BACK );
            }
            else
            {
                GL_ASSERT( glBindFramebuffer( GL_FRAMEBUFFER, m_qtPlz ) );
                GL_ASSERT( glDrawBuffers( 1, buffers ) );
            }

            GL_ASSERT( glClearColor( 0.0, 0.0, 0.0, 0.0 ) );
            // FIXME(Charly): Do we really need to clear the depth buffer ?
            GL_ASSERT( glClearDepth( 1.0 ) );
            GL_ASSERT( glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT ) );

            GL_ASSERT( glDepthFunc( GL_ALWAYS ) );

            GL_ASSERT( glViewport( 0, 0, m_width, m_height ) );

            const ShaderProgram* shader = m_drawScreenShader;
            shader->bind();
            shader->setUniform( "screenTexture", m_displayedTexture, 0 );
            m_quadMesh->render();

            GL_ASSERT( glDepthFunc( GL_LESS ) );
        }

        void Renderer::notifyRenderObjectsRenderingInternal()
        {
            for ( auto& ro : m_fancyRenderObjects )
            {
                ro->hasBeenRenderedOnce();
            }

            for ( auto& ro : m_debugRenderObjects )
            {
                ro->hasBeenRenderedOnce();
            }

            for ( auto& ro : m_xrayRenderObjects )
            {
                ro->hasBeenRenderedOnce();
            }

            for ( auto& ro : m_uiRenderObjects )
            {
                ro->hasBeenRenderedOnce();
            }
        }

        void Renderer::resize( uint w, uint h )
        {
            m_width = w;
            m_height = h;
            glViewport( 0, 0, m_width, m_height );

            m_depthTexture->Generate(m_width, m_height, GL_DEPTH_COMPONENT);
            m_pickingTexture->Generate(w, h, GL_RGBA_INTEGER);
            m_fancyTexture->Generate(w, h, GL_RGBA);

            m_pickingFbo->bind();
            m_pickingFbo->setSize( w, h );
            m_pickingFbo->attachTexture( GL_DEPTH_ATTACHMENT , m_depthTexture);
            m_pickingFbo->attachTexture( GL_COLOR_ATTACHMENT0, m_pickingTexture);
            m_pickingFbo->check();
            m_pickingFbo->unbind( true );

            GL_CHECK_ERROR;

            // Reset framebuffer state
            GL_ASSERT( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );

            GL_ASSERT( glDrawBuffer( GL_BACK ) );
            GL_ASSERT( glReadBuffer( GL_BACK ) );

            resizeInternal();
        }

        void Renderer::displayTexture( const std::string& texName )
        {
            if ( m_secondaryTextures.find( texName) != m_secondaryTextures.end() )
            {
                m_displayedTexture = m_secondaryTextures[texName];
            }
            else
            {
                m_displayedTexture = m_fancyTexture;
            }
        }

        std::vector<std::string> Renderer::getAvailableTextures() const
        {
            std::vector<std::string> ret;
            ret.push_back( "Fancy Texture" );
            for ( const auto& tex : m_secondaryTextures )
            {
                ret.push_back( tex.first );
            }
            return ret;
        }

        void Renderer::reloadShaders()
        {
            m_assetMgr->reloadShaderPrograms();
        }

        void Renderer::handleFileLoading( const std::string& filename )
        {
            Assimp::Importer importer;
            const aiScene* scene = importer.ReadFile( filename,
                                                      aiProcess_Triangulate |
                                                      aiProcess_JoinIdenticalVertices |
                                                      aiProcess_GenSmoothNormals |
                                                      aiProcess_SortByPType |
                                                      aiProcess_FixInfacingNormals |
                                                      aiProcess_CalcTangentSpace |
                                                      aiProcess_GenUVCoords );

            if ( !scene )
            {
                return;
            }

            if ( !scene->HasLights() )
            {
                return;
            }

            // Load lights
            for ( uint lightId = 0; lightId < scene->mNumLights; ++lightId )
            {
                aiLight* ailight = scene->mLights[lightId];

                aiString name = ailight->mName;
                aiNode* node = scene->mRootNode->FindNode( name );

                Core::Matrix4 transform( Core::Matrix4::Identity() );

                if ( node != nullptr )
                {
                    Core::Matrix4 t0;
                    Core::Matrix4 t1;

                    for ( uint i = 0; i < 4; ++i )
                    {
                        for ( uint j = 0; j < 4; ++j )
                        {
                            t0( i, j ) = scene->mRootNode->mTransformation[i][j];
                            t1( i, j ) = node->mTransformation[i][j];
                        }
                    }
                    transform = t0 * t1;
                }

                Core::Color color( ailight->mColorDiffuse.r,
                                   ailight->mColorDiffuse.g,
                                   ailight->mColorDiffuse.b, 1.0 );

                switch ( ailight->mType )
                {
                    case aiLightSource_DIRECTIONAL:
                    {
                        Core::Vector4 dir( ailight->mDirection[0],
                                           ailight->mDirection[1],
                                           ailight->mDirection[2], 0.0 );
                        dir = transform.transpose().inverse() * dir;

                        Core::Vector3 finalDir( dir.x(), dir.y(), dir.z() );
                        finalDir = -finalDir;

                        auto light = std::shared_ptr<DirectionalLight>( new DirectionalLight() );
                        light->setColor( color );
                        light->setDirection( finalDir );

                        addLight( light );

                    }
                    break;

                    case aiLightSource_POINT:
                    {
                        Core::Vector4 pos( ailight->mPosition[0],
                                           ailight->mPosition[1],
                                           ailight->mPosition[2], 1.0 );
                        pos = transform * pos;
                        pos /= pos.w();

                        auto light = std::shared_ptr<PointLight>( new PointLight() );
                        light->setColor( color );
                        light->setPosition( Core::Vector3( pos.x(), pos.y(), pos.z() ) );
                        light->setAttenuation( ailight->mAttenuationConstant,
                                               ailight->mAttenuationLinear,
                                               ailight->mAttenuationQuadratic );

                        addLight( light );

                    }
                    break;

                    case aiLightSource_SPOT:
                    {
                        Core::Vector4 pos( ailight->mPosition[0],
                                           ailight->mPosition[1],
                                           ailight->mPosition[2], 1.0 );
                        pos = transform * pos;
                        pos /= pos.w();

                        Core::Vector4 dir( ailight->mDirection[0],
                                           ailight->mDirection[1],
                                           ailight->mDirection[2], 0.0 );
                        dir = transform.transpose().inverse() * dir;

                        Core::Vector3 finalDir( dir.x(), dir.y(), dir.z() );
                        finalDir = -finalDir;

                        auto light = std::shared_ptr<SpotLight>( new SpotLight() );
                        light->setColor( color );
                        light->setPosition( Core::Vector3( pos.x(), pos.y(), pos.z() ) );
                        light->setDirection( finalDir );

                        light->setAttenuation( ailight->mAttenuationConstant,
                                               ailight->mAttenuationLinear,
                                               ailight->mAttenuationQuadratic );

                        light->setInnerAngleInRadians( ailight->mAngleInnerCone );
                        light->setOuterAngleInRadians( ailight->mAngleOuterCone );

                        addLight( light );

                    }
                    break;

                    case aiLightSource_UNDEFINED:
                    default:
                    {
                        //                LOG(ERROR) << "Light " << name.C_Str() << " has undefined type.";
                    } break;
                }
            }
        }

    }
} // namespace Ra
