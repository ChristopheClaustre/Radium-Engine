#include <GuiBase/Viewer/Viewer.hpp>

#include <iostream>

#include <QTimer>
#include <QMouseEvent>
#include <QPainter>

#include <Core/String/StringUtils.hpp>
#include <Core/Log/Log.hpp>
#include <Core/Math/ColorPresets.hpp>
#include <Core/Math/Math.hpp>
#include <Core/Containers/MakeShared.hpp>

#include <Core/Image/stb_image_write.h>

#include <Engine/Renderer/OpenGL/OpenGL.hpp>
#include <Engine/Component/Component.hpp>
#include <Engine/Renderer/Renderer.hpp>
#include <Engine/Renderer/Light/DirLight.hpp>
#include <Engine/Renderer/Camera/Camera.hpp>
#include <Engine/Renderer/Texture/Texture.hpp>
#include <Engine/Managers/SystemDisplay/SystemDisplay.hpp>
#include <Engine/Renderer/Renderers/ForwardRenderer.hpp>
#include <Engine/Renderer/Renderers/ExperimentalRenderer.hpp>

#include <GuiBase/Viewer/TrackballCamera.hpp>
#include <GuiBase/Viewer/Gizmo/GizmoManager.hpp>
#include <GuiBase/Utils/Keyboard.hpp>


namespace Ra
{
    Gui::Viewer::Viewer( QWidget* parent )
        : QOpenGLWidget( parent )
        , m_renderers(3)
        , m_gizmoManager(new GizmoManager(this))
        , m_renderThread( nullptr )
    {
        // Allow Viewer to receive events
        setFocusPolicy( Qt::StrongFocus );
        setMinimumSize( QSize( 800, 600 ) );

        m_camera.reset( new Gui::TrackballCamera( width(), height() ) );

        /// Intercept events to properly lock the renderer when it is compositing.

    }

    Gui::Viewer::~Viewer(){}

    void Gui::Viewer::initializeGL()
    {
        initializeOpenGLFunctions();

        LOG( logINFO ) << "*** Radium Engine Viewer ***";
        LOG( logINFO ) << "Renderer : " << glGetString( GL_RENDERER );
        LOG( logINFO ) << "Vendor   : " << glGetString( GL_VENDOR );
        LOG( logINFO ) << "OpenGL   : " << glGetString( GL_VERSION );
        LOG( logINFO ) << "GLSL     : " << glGetString( GL_SHADING_LANGUAGE_VERSION );

#if defined (OS_WINDOWS)
        glewExperimental = GL_TRUE;

        GLuint result = glewInit();
        if ( result != GLEW_OK )
        {
            std::string errorStr;
            Ra::Core::StringUtils::stringPrintf( errorStr, " GLEW init failed : %s", glewGetErrorString( result ) );
            CORE_ERROR( errorStr.c_str() );
        }
        else
        {
            LOG( logINFO ) << "GLEW     : " << glewGetString( GLEW_VERSION );
            glFlushError();
        }

#endif

        // FIXME(Charly): Renderer type should not be changed here
        //m_renderers.resize( 3 );
        // FIXME(Mathias): width and height might be wrong the first time ResizeGL is called (see QOpenGLWidget doc). This may cause problem on Retina display under MacOsX
        m_renderers[0].reset( new Engine::ForwardRenderer( width(), height() ) ); // Forward
        m_renderers[1].reset( nullptr ); // deferred
//        m_renderers[2].reset( new Engine::ExperimentalRenderer( width(), height() ) ); // experimental

        for ( auto& renderer : m_renderers )
        {
            if (renderer.get() != nullptr)
                renderer->initialize();
        }

        m_currentRenderer = m_renderers[0].get();

        m_light = Ra::Core::make_shared<Engine::DirectionalLight>();

        for ( auto& renderer : m_renderers )
        {
            if (renderer.get() != nullptr)
                renderer->addLight( m_light );
        }

        m_camera->attachLight( m_light );

        emit rendererReady();
    }

    Gui::CameraInterface* Gui::Viewer::getCameraInterface()
    {
        return m_camera.get();
    }

    Gui::GizmoManager* Gui::Viewer::getGizmoManager()
    {
        return m_gizmoManager;
    }

    const Engine::Renderer* Gui::Viewer::getRenderer() const
    {
        return m_currentRenderer;
    }

    void Gui::Viewer::onAboutToCompose()
    {
        // This slot function is called from the main thread as part of the event loop
        // when the GUI is about to update. We have to wait for the rendering to finish.
        m_currentRenderer->lockRendering();
    }

    void Gui::Viewer::onFrameSwapped()
    {
        // This slot is called from the main thread as part of the event loop when the
        // GUI has finished displaying the rendered image, so we unlock the renderer.
        m_currentRenderer->unlockRendering();
    }

    void Gui::Viewer::onAboutToResize()
    {
        // Like swap buffers, resizing is a blocking operation and we have to wait for the rendering
        // to finish before resizing.
        m_currentRenderer->lockRendering();
    }

    void Gui::Viewer::onResized()
    {
        m_currentRenderer->unlockRendering();
    }

    void Gui::Viewer::resizeGL( int width, int height )
    {
        // FIXME(Mathias) : Problem of glarea dimension on OsX Retina Display (half the size)
        // Renderer should have been locked by previous events.
        m_camera->resizeViewport( width, height );
        m_currentRenderer->resize( width, height );
    }

    void Gui::Viewer::mousePressEvent( QMouseEvent* event )
    {
        switch ( event->button() )
        {
            case Qt::LeftButton:
            {
#ifdef OS_MACOS
                // (Mathias) no middle button on Apple (only left, right and wheel)
                // replace middle button by <ctrl>+left (note : ctrl = "command"
                // fake the subsistem by setting MiddleButtonEvent and masking ControlModifier
                if (event->modifiers().testFlag( Qt::ControlModifier ) )
                {
                    auto mods = event->modifiers();
                    mods^=Qt::ControlModifier;
                    QMouseEvent macevent(event->type(), event->localPos(), event->windowPos(), event->screenPos(),
                                                    Qt::MiddleButton, event->buttons(),
                                                    mods, event->source() );
                    m_camera->handleMousePressEvent(&macevent);
                }
#endif
                if ( isKeyPressed( Qt::Key_Space ) )
                {
                    LOG( logINFO ) << "Raycast query launched";
                    Core::Ray r = m_camera->getCamera()->getRayFromScreen(Core::Vector2(event->x(), event->y()));
                    RA_DISPLAY_POINT(r.origin(), Core::Colors::Cyan(), 0.1f);
                    RA_DISPLAY_RAY(r, Core::Colors::Yellow());
                    auto ents = Engine::RadiumEngine::getInstance()->getEntityManager()->getEntities();
                    for (auto e : ents)
                    {
                        e->rayCastQuery(r);
                    }
                }
                else
                {
                    Engine::Renderer::PickingQuery query  = { Core::Vector2(event->x(), height() - event->y()), Core::MouseButton::RA_MOUSE_LEFT_BUTTON };
                    m_currentRenderer->addPickingRequest(query);
                    m_gizmoManager->handleMousePressEvent(event);
                }
            }
            break;

            case Qt::MiddleButton:
            {
                m_camera->handleMousePressEvent(event);
            }
            break;

            case Qt::RightButton:
            {
                // Check picking
                Engine::Renderer::PickingQuery query  = { Core::Vector2(event->x(), height() - event->y()), Core::MouseButton::RA_MOUSE_RIGHT_BUTTON };
                m_currentRenderer->addPickingRequest(query);
            }
            break;

            default:
            {
            } break;
        }
    }

    void Gui::Viewer::mouseReleaseEvent( QMouseEvent* event )
    {
        m_camera->handleMouseReleaseEvent( event );
        m_gizmoManager->handleMouseReleaseEvent(event);
    }

    void Gui::Viewer::mouseMoveEvent( QMouseEvent* event )
    {
        m_camera->handleMouseMoveEvent( event );
        m_gizmoManager->handleMouseMoveEvent(event);
    }

    void Gui::Viewer::wheelEvent( QWheelEvent* event )
    {
        m_camera->handleWheelEvent(event);
        QOpenGLWidget::wheelEvent( event );
    }

    void Gui::Viewer::keyPressEvent( QKeyEvent* event )
    {
        keyPressed(event->key());
        m_camera->handleKeyPressEvent( event );

        QOpenGLWidget::keyPressEvent(event);
    }

    void Gui::Viewer::keyReleaseEvent( QKeyEvent* event )
    {
        keyReleased(event->key());
        m_camera->handleKeyReleaseEvent( event );

        if (event->key() == Qt::Key_Z && !event->isAutoRepeat())
        {
            m_currentRenderer->toggleWireframe();
        }

        QOpenGLWidget::keyReleaseEvent(event);
    }

    void Gui::Viewer::reloadShaders()
    {
        // FIXME : check thread-saefty of this.
        m_currentRenderer->lockRendering();
        makeCurrent();
        m_currentRenderer->reloadShaders();
        doneCurrent();
        m_currentRenderer->unlockRendering();
    }

    void Gui::Viewer::displayTexture( const QString &tex )
    {
        m_currentRenderer->lockRendering();
        m_currentRenderer->displayTexture( tex.toStdString() );
        m_currentRenderer->unlockRendering();
    }

    void Gui::Viewer::changeRenderer( int index )
    {
        if (m_renderers[index] != nullptr) {
            // NOTE(Charly): This is probably buggy since it has not been tested.
            LOG( logWARNING ) << "Changing renderers might be buggy since it has not been tested.";

            m_currentRenderer->lockRendering();
            m_currentRenderer = m_renderers[index].get();
            m_currentRenderer->initialize();
            m_currentRenderer->resize( width(), height() );
            emit rendererReady();
            m_currentRenderer->unlockRendering();
        }
    }

    int Gui::Viewer::addRenderer( Engine::Renderer * renderer )
    {
        if (renderer != nullptr)
        {
            // Tested by the "dynamic sampling and rendering of algebraic point set surfaces" students team
            // needed if an plugin want to add an renderer
            int index = m_renderers.size();
            m_renderers.emplace(m_renderers.end(), renderer);

            renderer->addLight( m_light );

            return index;
        }
        else
        {
            LOG(logWARNING) << "Impossible to add the renderer to the list of renderers.";
            return -1;
        }
    }

    // Asynchronous rendering implementation

    void Gui::Viewer::startRendering( const Scalar dt )
    {
        makeCurrent();

        // Move camera if needed. Disabled for now as it takes too long (see issue #69)
        //m_camera->update( dt );

        Engine::RenderData data;
        data.dt = dt;
        data.projMatrix = m_camera->getProjMatrix();
        data.viewMatrix = m_camera->getViewMatrix();
        m_currentRenderer->render( data );
    }

    void Gui::Viewer::waitForRendering()
    {
    }

    void Gui::Viewer::handleFileLoading( const std::string& file )
    {
        for ( auto& renderer : m_renderers )
        {
            if (renderer.get() != nullptr)
                renderer->handleFileLoading( file );
        }
    }

    void Gui::Viewer::processPicking()
    {
        CORE_ASSERT( m_currentRenderer->getPickingQueries().size() == m_currentRenderer->getPickingResults().size(),
                    "There should be one result per query." );

        for (uint i = 0 ; i < m_currentRenderer->getPickingQueries().size(); ++i)
        {
            const Engine::Renderer::PickingQuery& query  = m_currentRenderer->getPickingQueries()[i];
            if ( query.m_button == Core::MouseButton::RA_MOUSE_LEFT_BUTTON)
            {
                emit leftClickPicking(m_currentRenderer->getPickingResults()[i]);
            }
            else if (query.m_button == Core::MouseButton::RA_MOUSE_RIGHT_BUTTON)
            {
                emit rightClickPicking(m_currentRenderer->getPickingResults()[i]);
            }
        }
    }

    void Gui::Viewer::fitCameraToScene( const Core::Aabb& aabb )
    {
        if (!aabb.isEmpty())
        {
            m_camera->fitScene(aabb);
        }
    }

    std::vector<std::string> Gui::Viewer::getRenderersName() const
    {
        std::vector<std::string> ret;

        for ( const auto& renderer : m_renderers )
        {
            if (renderer.get() != nullptr)
                ret.push_back( renderer->getRendererName() );
        }

        return ret;
    }

    void Gui::Viewer::grabFrame( const std::string& filename )
    {
        makeCurrent();

        Engine::Texture* tex = m_currentRenderer->getDisplayTexture();
        tex->bind();

        // Get a buffer to store the pixels of the OpenGL texture (in float format)
        float* pixels = new float[tex->width() * tex->height() * 4];

        // Grab the texture data
        GL_ASSERT(glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pixels));

        // Now we must convert the floats to RGB while flipping the image updisde down.
        uchar* writtenPixels = new uchar[tex->width() * tex->height() * 4];
        for (uint j = 0; j < tex->height(); ++j)
        {
            for (uint i = 0; i < tex->width(); ++i)
            {
                uint in = 4 * (j * tex->width() + i);  // Index in the texture buffer
                uint ou = 4 * ((tex->height() - 1 - j) * tex->width() + i); // Index in the final image (note the j flipping).

                writtenPixels[ou + 0] = (uchar)Ra::Core::Math::clamp<Scalar>(pixels[in + 0] * 255.f, 0, 255);
                writtenPixels[ou + 1] = (uchar)Ra::Core::Math::clamp<Scalar>(pixels[in + 1] * 255.f, 0, 255);
                writtenPixels[ou + 2] = (uchar)Ra::Core::Math::clamp<Scalar>(pixels[in + 2] * 255.f, 0, 255);
                writtenPixels[ou + 3] = 0xff;
            }
        }

        std::string ext = Core::StringUtils::getFileExt(filename);

        if (ext == "bmp")
        {
            stbi_write_bmp(filename.c_str(), tex->width(), tex->height(), 4, writtenPixels);
        }
        else if (ext == "png")
        {
            stbi_write_png(filename.c_str(), tex->width(), tex->height(), 4, writtenPixels, tex->width() * 4 * sizeof(uchar));
        }
        else
        {
            LOG(logWARNING) << "Cannot write frame to "<<filename<<" : unsupported extension";
        }

        delete[] pixels;
        delete[] writtenPixels;
    }

    void Gui::Viewer::enablePostProcess(int enabled)
    {
        m_currentRenderer->enablePostProcess(enabled);
    }

    void Gui::Viewer::enableDebugDraw(int enabled)
    {
        m_currentRenderer->enableDebugDraw(enabled);
    }
} // namespace Ra
