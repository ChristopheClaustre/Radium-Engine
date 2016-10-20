#include <MainApplication/Viewer/Viewer.hpp>

#include <iostream>

#include <QTimer>
#include <QMouseEvent>
#include <QPainter>

#include <Core/String/StringUtils.hpp>
#include <Core/Log/Log.hpp>
#include <Core/Math/ColorPresets.hpp>
#include <Core/Math/Math.hpp>
#include <Core/Containers/MakeShared.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <Core/Image/stb_image_write.h>

#include <Engine/Renderer/OpenGL/OpenGL.hpp>
#include <Engine/Component/Component.hpp>
#include <Engine/Renderer/Renderer.hpp>
#include <Engine/Renderer/Light/DirLight.hpp>
#include <Engine/Renderer/Camera/Camera.hpp>
#include <Engine/Renderer/Texture/Texture.hpp>
#include <Engine/Managers/SystemDisplay/SystemDisplay.hpp>
#include <Engine/Renderer/Renderers/ForwardRenderer.hpp>

#include <MainApplication/Viewer/TrackballCamera.hpp>
#include <MainApplication/Viewer/Gizmo/GizmoManager.hpp>
#include <MainApplication/Gui/MainWindow.hpp>
#include <MainApplication/MainApplication.hpp>
#include <MainApplication/Utils/Keyboard.hpp>

namespace Ra
{
    Gui::Viewer::Viewer( QWidget* parent )
        : QOpenGLWidget( parent )
        , m_gizmoManager(new GizmoManager(this))
    {
        // Allow Viewer to receive events
        setFocusPolicy( Qt::StrongFocus );
        setMinimumSize( QSize( 800, 600 ) );

        m_camera.reset( new Gui::TrackballCamera( width(), height() ) );
    }

    Gui::Viewer::~Viewer()
    {
    }

    void Gui::Viewer::initializeGL()
    {
        if (gl3wInit())
        {
            CORE_ERROR("Could not get a valid OpenGL3.0 context.");
        }

#if defined(USE_OPENGL330)
        if (!gl3wIsSupported(3, 2))
        {
            CORE_ERROR("OpenGL 3.2 is not supported.");
        }
#else
        if (!gl3wIsSupported(4, 1))
        {
            CORE_ERROR("OpenGL 4.1 is not supported.");
        }
#endif

        LOG( logINFO ) << "*** Radium Engine Viewer ***";
        LOG( logINFO ) << "Renderer : " << glGetString( GL_RENDERER );
        LOG( logINFO ) << "Vendor   : " << glGetString( GL_VENDOR );
        LOG( logINFO ) << "OpenGL   : " << glGetString( GL_VERSION );
        LOG( logINFO ) << "GLSL     : " << glGetString( GL_SHADING_LANGUAGE_VERSION );

        // FIXME(Charly): Renderer type should not be changed here
        m_renderers.resize( 1 );
        // FIXME(Mathias): width and height might be wrong the first time ResizeGL is called (see QOpenGLWidget doc). This may cause problem on Retina display under MacOsX
        m_renderers[0].reset( new Engine::ForwardRenderer( width(), height() ) );

        for ( auto& renderer : m_renderers )
        {
            renderer->initialize();
        }

        m_currentRenderer = m_renderers[0].get();

        // FIXME (Mathias) : according to modern C++ guidelines (Stroustrup), prefer the following
        // NOTE(Charly) : Indeed, but on MSVC std::make_shared does not guarantee alignement, hence making Eigen crash.
        //                We introduced Ra::Core::make_shared later and I still have to change all this calls.
        auto light = Ra::Core::make_shared<Engine::DirectionalLight>();

        for ( auto& renderer : m_renderers )
        {
            renderer->addLight( light );
        }

        m_camera->attachLight( light );

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

    void Gui::Viewer::resizeGL( int width, int height )
    {
        makeCurrent();

        LOG(logINFO) << "Resize";
        GL_CHECK_ERROR;
        GLuint texture;
        GL_ASSERT(glGenTextures(1, &texture));

        // FIXME(Mathias) : Problem of glarea dimension on OsX Retina Display (half the size)
        // Renderer should have been locked by previous events.
        m_camera->resizeViewport( width, height );
        m_currentRenderer->resize( width, height );

        doneCurrent();
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
                    auto macevent = new QMouseEvent(event->type(), event->localPos(), event->windowPos(), event->screenPos(),
                                                    Qt::MiddleButton, event->buttons(),
                                                    mods, event->source() );
                    m_camera->handleMousePressEvent(macevent);
                    delete macevent;
                }
#endif
                if ( isKeyPressed( Key_Space ) )
                {
                    LOG( logINFO ) << "Raycast query launched";
                    Core::Ray r = m_camera->getCamera()->getRayFromScreen(Core::Vector2(event->x(), event->y()));
                    RA_DISPLAY_POINT(r.origin(), Core::Colors::Cyan(), 0.1f);
                    RA_DISPLAY_RAY(r, Core::Colors::Yellow());
                    auto ents = mainApp->getEngine()->getEntityManager()->getEntities();
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
        m_camera->handleKeyPressEvent( event );

        QOpenGLWidget::keyPressEvent(event);
    }

    void Gui::Viewer::keyReleaseEvent( QKeyEvent* event )
    {
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
        // NOTE(Charly): This is probably buggy since it has not been tested.
        LOG( logWARNING ) << "Changing renderers might be buggy since it has not been tested.";
        m_currentRenderer->lockRendering();
        m_currentRenderer = m_renderers[index].get();
        m_currentRenderer->initialize();
        m_currentRenderer->resize( width(), height() );
        m_currentRenderer->unlockRendering();
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

        doneCurrent();
    }

    void Gui::Viewer::waitForRendering()
    {
    }

    void Gui::Viewer::handleFileLoading( const std::string& file )
    {
        for ( auto& renderer : m_renderers )
        {
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
        // FIXME(Charly): Does not work, the camera needs to be fixed
        m_camera->fitScene( aabb );
    }

    std::vector<std::string> Gui::Viewer::getRenderersName() const
    {
        std::vector<std::string> ret;

        for ( const auto& r : m_renderers )
        {
            ret.push_back( r->getRendererName() );
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
        unsigned char* writtenPixels = new uchar[tex->width() * tex->height() * 4];
        for (uint j = 0; j < tex->height(); ++j)
        {
            for (uint i = 0; i < tex->width(); ++i)
            {
                uint in = 4 * (j * tex->width() + i);  // Index in the texture buffer
                uint ou = 4 * ((tex->height() - 1 - j) * tex->width() + i); // Index in the final image (note the j flipping).

                writtenPixels[ou + 0] = Ra::Core::Math::clamp<uchar>(pixels[in + 0] * 255.f, 0u, 0xffu);
                writtenPixels[ou + 1] = Ra::Core::Math::clamp<uchar>(pixels[in + 1] * 255.f, 0u, 0xffu);
                writtenPixels[ou + 2] = Ra::Core::Math::clamp<uchar>(pixels[in + 2] * 255.f, 0u, 0xffu);
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
