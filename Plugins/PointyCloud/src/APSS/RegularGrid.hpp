#ifndef POINTYCLOUDPLUGIN_REGULARGRID_HPP
#define POINTYCLOUDPLUGIN_REGULARGRID_HPP

#include <Core/Math/LinearAlgebra.hpp>

#include <vector>
#include <memory>

namespace PointyCloudPlugin {

    class RegularGrid
    {
        friend std::unique_ptr<RegularGrid> std::make_unique<RegularGrid>();
        friend class RegularGridBuilder;

    private:
        struct Cell {
            int index;
            int length;
        };

    public:
        ~RegularGrid();

        std::vector<int> query(const Ra::Core::Vector3& p) const;

    protected:
        RegularGrid();

        Ra::Core::Aabb m_aabb;

        std::vector<int> m_indices;

        std::vector<Cell> m_leaves;

    }; // class RegularGrid

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_REGULARGRID_HPP
