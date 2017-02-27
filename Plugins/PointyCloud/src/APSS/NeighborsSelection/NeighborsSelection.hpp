#ifndef POINTYCLOUDPLUGIN_NEIGHBORSSELECTION_HPP
#define POINTYCLOUDPLUGIN_NEIGHBORSSELECTION_HPP

#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <APSS/PointyCloud.hpp>

namespace PointyCloudPlugin
{

    class NeighborsProcessor;

    class NeighborsSelection
    {
    public:
        NeighborsSelection(std::shared_ptr<PointyCloud> cloud, const Scalar influenceRadius);
        virtual ~NeighborsSelection();

        virtual void getNeighbors(const APoint& point, std::vector<int> & indexSelected) const;
        virtual void processNeighbors(const APoint& point, NeighborsProcessor& f) const;
        virtual bool isEligible(const APoint& point) const;

        void setInfluenceRadius(Scalar influenceRadius) { m_influenceRadius = influenceRadius; }
    protected:

        std::shared_ptr<PointyCloud> m_cloud;
        Scalar m_influenceRadius;

    }; // class NeighborsSelection

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_NEIGHBORSSELECTION_HPP

