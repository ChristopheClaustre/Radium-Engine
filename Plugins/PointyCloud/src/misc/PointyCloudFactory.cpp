#include "PointyCloudFactory.hpp"

#include <APSS/PointyCloud.hpp>

namespace PointyCloudPlugin {

std::shared_ptr<PointyCloud> PointyCloudFactory::makeDenseCube(int n, double dl)
{
    std::shared_ptr<PointyCloud> cloud = std::make_shared<PointyCloud>();

    size_t size = n*n*n;

    cloud->m_points.resize(size);

    int p = 0;

    Ra::Core::Vector3 pos;
    Ra::Core::Vector3 nor;
    nor << 0, 0, 1;
    Ra::Core::Vector4 col;
    for(int i = 0; i<n; ++i)
    {
        pos[0] = i*dl;
        col[0] = (float)i/(n-1);
        for(int j = 0; j<n; ++j)
        {
            pos[1] = j*dl;
            col[1] = (float)j/(n-1);
            for(int k = 0; k<n; ++k, ++p)
            {
                pos[2] = k*dl;
                col[2] = (float)k/(n-1);
                cloud->m_points[p].pos() = pos;
                cloud->m_points[p].normal() = nor;
                cloud->m_points[p].color() = col;
            }
        }
    }

    return cloud;
}

}

