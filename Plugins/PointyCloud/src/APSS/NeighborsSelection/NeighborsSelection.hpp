#ifndef POINTYCLOUDPLUGIN_NEIGHBORSSELECTION_HPP
#define POINTYCLOUDPLUGIN_NEIGHBORSSELECTION_HPP

#include <Core/CoreMacros.hpp>

#include <memory>
#include <vector>

namespace PointyCloudPlugin
{
    class PointyCloud;
    class APoint;

    class NeighborsSelection
    {
    public:
        NeighborsSelection(std::shared_ptr<PointyCloud> cloud, const Scalar influenceRadius);
        ~NeighborsSelection();

        virtual std::vector<int> getNeighbors(const APoint& point) const;
        virtual bool isEligible(const APoint& point) const;

        void setInfluenceRadius(Scalar influenceRadius) { m_influenceRadius = influenceRadius; }
    protected:

        std::shared_ptr<PointyCloud> m_cloud;
        Scalar m_influenceRadius;

    }; // class NeighborsSelection

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_NEIGHBORSSELECTION_HPP

