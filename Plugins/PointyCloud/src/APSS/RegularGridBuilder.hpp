#ifndef POINTYCLOUDPLUGIN_REGULARGRIDBUILDER_HPP
#define POINTYCLOUDPLUGIN_REGULARGRIDBUILDER_HPP

#include <memory>

namespace PointyCloudPlugin {

    class RegularGrid;
    class PointyCloud;

    class RegularGridBuilder
    {
    public:

        static std::unique_ptr<RegularGrid> buildRegularGrid(const PointyCloud& cloud, double influenceRadius);

    }; // class RegularGrid

} // namespace RegularGridBuilder

#endif // POINTYCLOUDPLUGIN_REGULARGRIDBUILDER_HPP
