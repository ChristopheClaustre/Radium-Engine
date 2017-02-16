#ifndef POINTYCLOUDPLUGIN_NEIGHBORSSELECTION_HPP
#define POINTYCLOUDPLUGIN_NEIGHBORSSELECTION_HPP

#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <APSS/PointyCloud.hpp>

namespace PointyCloudPlugin
{

    class NeighborsSelection
    {
    public:
        NeighborsSelection(std::shared_ptr<PointyCloud> cloud, const Scalar influenceRadius);
        ~NeighborsSelection();

        std::vector<int> getNeighbors(APoint point);

        void setInfluenceRadius(Scalar influenceRadius) { m_influenceRadius = influenceRadius; }
    protected:

        std::shared_ptr<PointyCloud> m_cloud;
        Scalar m_influenceRadius;

    }; // class NeighborsSelection

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_NEIGHBORSSELECTION_HPP

