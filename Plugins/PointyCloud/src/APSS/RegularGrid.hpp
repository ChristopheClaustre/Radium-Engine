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
            Cell() : index(0), length(0) {}
            int index;
            int length;
        };

    public:
        ~RegularGrid();

        std::vector<int> query(const Ra::Core::Vector3& p, float r) const;

        void printAll() const;
        void printGrid() const;

    protected:
        RegularGrid();

        inline int rawIndex(int i, int j, int k) const {
            return k*(m_nx*m_ny) + j*m_nx + i;
        }

        inline int rawIndex(const Ra::Core::Vector3& p) const {
            return rawIndexLocal(p-m_aabb.min());
        }

        inline int rawIndexLocal(const Ra::Core::Vector3& pLocal) const {
            return rawIndex(std::floor(pLocal[0]/m_dx),
                            std::floor(pLocal[1]/m_dy),
                            std::floor(pLocal[2]/m_dz));
        }

        Ra::Core::Aabb m_aabb;

        float m_dx, m_dy, m_dz;

        int m_nx, m_ny, m_nz;

        std::vector<int> m_indices;

        std::vector<Cell> m_cells;

    }; // class RegularGrid

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_REGULARGRID_HPP
