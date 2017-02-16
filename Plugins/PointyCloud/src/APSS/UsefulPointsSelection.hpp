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
        UsefulPointsSelection(std::shared_ptr<PointyCloud> originalCloud, const Ra::Engine::Camera* camera);
        ~UsefulPointsSelection();

        PointyCloud selectUsefulPoints();

    protected:
        std::shared_ptr<PointyCloud> m_originalCloud;
        const Ra::Engine::Camera* m_camera;

    }; // class UsefulPointsSelection

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_USEFULPOINTSSELECTION_HPP
