#include "RegularGridBuilder.hpp"

#include "RegularGrid.hpp"
#include "PointyCloud.hpp"

namespace PointyCloudPlugin {

std::unique_ptr<RegularGrid> RegularGridBuilder::buildRegularGrid(const PointyCloud &cloud, double influenceRadius)
{
    std::unique_ptr<RegularGrid> grid = std::make_unique<RegularGrid>();

    //TODO

    return grid;
}

} // namespace PointyCloudPlugin
