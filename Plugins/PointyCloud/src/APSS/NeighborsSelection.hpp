#ifndef POINTYCLOUDPLUGIN_NEIGHBORSSELECTION_HPP
#define POINTYCLOUDPLUGIN_NEIGHBORSSELECTION_HPP

#include <Engine/Renderer/Mesh/Mesh.hpp>

namespace PointyCloudPlugin
{

    class NeighborsSelection
    {
    public:
        NeighborsSelection(std::shared_ptr<Ra::Engine::Mesh> cloud,const float influenceRadius);
        ~NeighborsSelection();

        std::vector<int> getNeighbors(Ra::Core::Vector3 vertexPosition);
    protected:

        std::shared_ptr<Ra::Engine::Mesh> m_cloud;
        float m_influenceRadius;

    }; // class NeighborsSelection

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_NEIGHBORSSELECTION_HPP

