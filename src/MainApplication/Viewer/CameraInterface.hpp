#ifndef RADIUMENGINE_CAMERAINTERFACE_HPP
#define RADIUMENGINE_CAMERAINTERFACE_HPP

#include <memory>

#include <QObject>
#include <QKeyEvent>
#include <QMouseEvent>

#include <Core/CoreMacros.hpp>
#include <Core/Math/Vector.hpp>
#include <Core/Math/Matrix.hpp>

namespace Ra { namespace Core   { struct MouseEvent; } }
namespace Ra { namespace Core   { struct KeyEvent; } }
namespace Ra { namespace Engine { class  Camera; } }

namespace Ra { namespace Gui {

class CameraInterface : public QObject
{
    Q_OBJECT

public:
    // FIXME(Charly): width / height ?
    CameraInterface(uint width, uint height);
    virtual ~CameraInterface();

    void resizeViewport(uint width, uint height);

    Core::Matrix4 getProjMatrix() const;
    Core::Matrix4 getViewMatrix() const;


    /// @return true if the event has been taken into account, false otherwise
    virtual bool handleMousePressEvent(QMouseEvent* event) = 0;
    /// @return true if the event has been taken into account, false otherwise
    virtual bool handleMouseReleaseEvent(QMouseEvent* event) = 0;
    /// @return true if the event has been taken into account, false otherwise
    virtual bool handleMouseMoveEvent(QMouseEvent* event) = 0;

    /// @return true if the event has been taken into account, false otherwise
    virtual bool handleKeyPressEvent(QKeyEvent* event) = 0;
    /// @return true if the event has been taken into account, false otherwise
    virtual bool handleKeyReleaseEvent(QKeyEvent* event) = 0;

public slots:
    void setCameraSensitivity(Scalar sensitivity);

    void setCameraFov(Scalar fov);
    void setCameraFovInDegrees(Scalar fov);
    void setCameraZNear(Scalar zNear);
    void setCameraZFar(Scalar zFar);

    void mapCameraBehaviourToAabb(const Core::Aabb& aabb);
    void unmapCameraBehaviourToAabb();

    virtual void moveCameraToFitAabb(const Core::Aabb& aabb);
    virtual void moveCameraTo(const Core::Vector3& position);

    virtual void resetCamera() = 0;

protected:
    Core::Aabb m_targetedAabb;

    Scalar m_targetedAabbVolume;
    Scalar m_cameraSensitivity;

    uint m_width;
    uint m_height;

    mutable std::unique_ptr<Engine::Camera> m_camera;
    mutable bool m_viewIsDirty;
    mutable bool m_projIsDirty;
    bool m_mapCameraBahaviourToAabb;
};

} // namespace Ra
} // namespace Engine

#endif // RADIUMENGINE_CAMERAINTERFACE_HPP