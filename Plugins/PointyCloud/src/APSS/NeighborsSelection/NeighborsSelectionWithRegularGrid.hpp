#ifndef POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONWITHREGULARGRID_HPP
#define POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONWITHREGULARGRID_HPP

#include <APSS/NeighborsSelection/NeighborsSelection.hpp>

namespace PointyCloudPlugin {

    class RegularGrid;

    class NeighborsSelectionWithRegularGrid : public NeighborsSelection
    {
    public:
        NeighborsSelectionWithRegularGrid(std::shared_ptr<PointyCloud> cloud, float influenceRadius, int nCell = 100);
        ~NeighborsSelectionWithRegularGrid();

        virtual std::vector<int> getNeighbors(const APoint &point) const;
        virtual bool isEligible(const APoint& point) const;

        const RegularGrid* grid() const;

    protected:

        std::unique_ptr<RegularGrid> m_grid;

    }; // class NeighborsSelectionWithRegularGrid

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONWITHREGULARGRID_HPP