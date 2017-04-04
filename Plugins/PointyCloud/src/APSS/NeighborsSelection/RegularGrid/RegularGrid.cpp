#include "RegularGrid.hpp"

#include <APSS/PointyCloud.hpp>
#include <APSS/NeighborsSelection/NeighborsProcessor.hpp>

#include <Core/Log/Log.hpp>
#include <sstream>

namespace PointyCloudPlugin {

RegularGrid::RegularGrid()
{
}

RegularGrid::~RegularGrid()
{
}

void RegularGrid::query(const Ra::Core::Vector3& p, float r, std::vector<int> & indices) const
{
    // point in local coordinates
    Ra::Core::Vector3 q = p - m_aabb.min();

    // searching limits of a cube centered at q of length r
    int imin = std::max((int)std::floor((q[0]-r)/m_dx), 0);
    int jmin = std::max((int)std::floor((q[1]-r)/m_dy), 0);
    int kmin = std::max((int)std::floor((q[2]-r)/m_dz), 0);
    int imax = std::min((int)std::floor((q[0]+r)/m_dx), m_nx-1);
    int jmax = std::min((int)std::floor((q[1]+r)/m_dy), m_ny-1);
    int kmax = std::min((int)std::floor((q[2]+r)/m_dz), m_nz-1);

    // search
    for(int k = kmin; k<=kmax; ++k)
    {
        for(int j = jmin; j<=jmax; ++j)
        {
            for(int i = imin; i<=imax; ++i)
            {
                int idxCell = rawIndex(i, j, k);
                int begin = m_cells[idxCell].index;
                int length = m_cells[idxCell].length;

                for(int idx = begin; idx<begin+length; ++idx)
                {
                    //TODO it may be faster to avoid push_back and use remove_if ?
                    if((p - m_cloud->at(m_indices[idx]).pos()).norm() <= r) {
                        indices.push_back(m_indices[idx]);
                    }
                }
            }
        }
    }
}

void RegularGrid::process(const Ra::Core::Vector3& p, float r, NeighborsProcessor& f) const
{
    // point in local coordinates
    Ra::Core::Vector3 q = p - m_aabb.min();

    // searching limits of a cube centered at q of length r
    int imin = std::max((int)std::floor((q[0]-r)/m_dx), 0);
    int jmin = std::max((int)std::floor((q[1]-r)/m_dy), 0);
    int kmin = std::max((int)std::floor((q[2]-r)/m_dz), 0);
    int imax = std::min((int)std::floor((q[0]+r)/m_dx), m_nx-1);
    int jmax = std::min((int)std::floor((q[1]+r)/m_dy), m_ny-1);
    int kmax = std::min((int)std::floor((q[2]+r)/m_dz), m_nz-1);

    // search
    for(int k = kmin; k<=kmax; ++k)
    {
        for(int j = jmin; j<=jmax; ++j)
        {
            for(int i = imin; i<=imax; ++i)
            {
                int idxCell = rawIndex(i, j, k);
                int begin = m_cells[idxCell].index;
                int length = m_cells[idxCell].length;

                for(int idx = begin; idx<begin+length; ++idx)
                {
                    //TODO it may be faster to avoid push_back and use remove_if ?
                    if((p - m_cloud->m_points[m_indices[idx]].pos()).norm() <= r) {
                        f(m_indices[idx]);
                    }
                }
            }
        }
    }
}

bool RegularGrid::hasNeighbors(const Ra::Core::Vector3& p, float r) const
{
    // point in local coordinates
    Ra::Core::Vector3 q = p - m_aabb.min();

    // searching limits of a cube centered at q of length r
    int imin = std::max((int)std::floor((q[0]-r)/m_dx), 0);
    int jmin = std::max((int)std::floor((q[1]-r)/m_dy), 0);
    int kmin = std::max((int)std::floor((q[2]-r)/m_dz), 0);
    int imax = std::min((int)std::floor((q[0]+r)/m_dx), m_nx-1);
    int jmax = std::min((int)std::floor((q[1]+r)/m_dy), m_ny-1);
    int kmax = std::min((int)std::floor((q[2]+r)/m_dz), m_nz-1);

    // neighbors count
    int res = 0;

    bool enough = false;
    // search
    int k = kmin;
    while(k<=kmax)
    {
        int j = jmin;
        while(j<=jmax)
        {
            int i = imin;
            while(i<=imax)
            {
                int idxCell = rawIndex(i, j, k);
                int begin = m_cells[idxCell].index;
                int length = m_cells[idxCell].length;

                int idx = begin;
                while(idx<begin+length)
                {
                    if((p - m_cloud->at(m_indices[idx]).pos()).norm() <= r)
                    {
                        ++res;
                    }
                    ++idx;
                }

                // is it enough ?? (cf. Patate)
                enough = (res>=6);

                ++i;
            }
            ++j;
        }
        ++k;
    }

    return enough;
}

float RegularGrid::getBuildTime() const {
    return m_buildTime;
}

float RegularGrid::getDx() const {
    return m_dx;
}

float RegularGrid::getDy() const {
    return m_dy;
}

float RegularGrid::getDz() const {
    return m_dz;
}

int RegularGrid::getNx() const {
    return m_nx;
}

int RegularGrid::getNy() const {
    return m_ny;
}

int RegularGrid::getNz() const {
    return m_nz;
}

void RegularGrid::printAll() const
{
    std::stringstream output;
    output << "\nAabb = [" << m_aabb.min()[0] << "," << m_aabb.min()[1] << "," << m_aabb.min()[2] << "];[" <<
              m_aabb.max()[0] << "," << m_aabb.max()[1] << "," << m_aabb.max()[2] << "\n";
    output << "dx=" << m_dx << " " << "dy=" << m_dy << " " << "dz=" << m_dz << "\n";
    output << "nx=" << m_nx << " " << "ny=" << m_ny << " " << "nz=" << m_nz << "\n";

    LOG(logINFO) << "Regular grid :\n";
    LOG(logINFO) << output.str();

    printGrid();
}

void RegularGrid::printGrid() const
{
    std::stringstream output;
    int k = 0;
    for(const auto& cell : m_cells)
    {
        output << k++ <<":["<< cell.index <<","<< cell.length<<"] -> ";
        for(int i = cell.index; i < cell.index+cell.length; ++i)
            output << m_indices[i] << " ";
        output << "\n";
    }

    output << "\n";
    for(const auto& idx : m_indices)
        output << idx << " ";

    LOG(logINFO) << "\n" << output.str() << "\n";
}

} // namespace PointyCloudPlugin
