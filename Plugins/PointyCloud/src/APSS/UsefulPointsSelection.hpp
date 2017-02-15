#ifndef POINTYCLOUDPLUGIN_USEFULPOINTSSELECTION_HPP
#define POINTYCLOUDPLUGIN_USEFULPOINTSSELECTION_HPP

//#include <memory>
#include <vector>
#include <Engine/Renderer/Camera/Camera.hpp>

#include <APSS/PointyCloud.hpp>

namespace PointyCloudPlugin
{
    class UsefulPointsSelection
    {
    public:
        UsefulPointsSelection(PointyCloud cloud, const Ra::Engine::Camera* camera);
        ~UsefulPointsSelection();

        PointyCloud selectUsefulPoints();

    protected:
        const PointyCloud m_cloud;
        const Ra::Engine::Camera* m_camera;

    }; // class UsefulPointsSelection

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_USEFULPOINTSSELECTION_HPP
