#ifndef POINTYCLOUDPLUGIN_REGULARGRIDBUILDER_HPP
#define POINTYCLOUDPLUGIN_REGULARGRIDBUILDER_HPP

#include <Core/Math/LinearAlgebra.hpp>

#include <memory>

namespace PointyCloudPlugin {

    class RegularGrid;
    class PointyCloud;

    class RegularGridBuilder
    {
    public:

        static std::unique_ptr<RegularGrid> buildRegularGrid(std::shared_ptr<PointyCloud> cloud, double influenceRadius);

    protected:

        static Ra::Core::Aabb computeAabb(std::shared_ptr<PointyCloud> cloud);

    }; // class RegularGrid

} // namespace RegularGridBuilder

#endif // POINTYCLOUDPLUGIN_REGULARGRIDBUILDER_HPP
