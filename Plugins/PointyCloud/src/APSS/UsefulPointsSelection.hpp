#ifndef POINTYCLOUDPLUGIN_USEFULPOINTSSELECTION_HPP
#define POINTYCLOUDPLUGIN_USEFULPOINTSSELECTION_HPP

#include <memory>
#include <vector>
#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <Engine/Renderer/Camera/Camera.hpp>

namespace PointyCloudPlugin
{

    class UsefulPointsSelection
    {
    public:
        UsefulPointsSelection(const Ra::Engine::Mesh* cloud, const Ra::Engine::Camera* camera);
        ~UsefulPointsSelection();

        std::vector<int> selectUsefulPoints() const;

    protected:

        bool isUsefulPoint(int idx) const;

        const Ra::Engine::Mesh* m_cloud;

        const Ra::Engine::Camera* m_camera;

    }; // class UsefulPointsSelection

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_USEFULPOINTSSELECTION_HPP
