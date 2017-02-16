#ifndef POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONWITHREGULARGRID_HPP
#define POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONWITHREGULARGRID_HPP

#include "NeighborsSelection.hpp"

namespace PointyCloudPlugin {

    class RegularGrid;

    class NeighborsSelectionWithRegularGrid : public NeighborsSelection
    {
    public:
        NeighborsSelectionWithRegularGrid(std::shared_ptr<Ra::Engine::Mesh> cloud, float influenceRadius);
        ~NeighborsSelectionWithRegularGrid();

    protected:

        std::unique_ptr<RegularGrid> m_grid;

    }; // class NeighborsSelectionWithRegularGrid

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONWITHREGULARGRID_HPP
