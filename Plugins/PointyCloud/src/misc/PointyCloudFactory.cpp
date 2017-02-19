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
    Ra::Core::Vector4 col;
    col[3] = 1.0;
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
                cloud->m_points[p].normal() = Ra::Core::Vector3::Ones().normalized();
                cloud->m_points[p].color() = col;
            }
        }
    }

    return cloud;
}

std::shared_ptr<PointyCloud> PointyCloudFactory::makeSphere(double radius, int n, int m)
{
    std::shared_ptr<PointyCloud> cloud = std::make_shared<PointyCloud>();

    size_t size = n*(m-2)+2; // -2/+2 for poles

    cloud->m_points.resize(size);

    double dTheta = 2.0*M_PI/n;
    double dPhi = M_PI/(m-1);

    Ra::Core::Vector3 pos;
    Ra::Core::Vector3 nor;
    Ra::Core::Vector4 col;
    col[2] = 0.5;

    int p = 0;

    for(int i = 0; i<n; ++i)
    {
        double theta = i*dTheta;
        col[0] = (float)i/(n-1);
        for(int j = 1; j<m-1; ++j, ++p)
        {
            double phi = j*dPhi;
            col[1] = (float)j/(m-1);

            pos[0] = radius*std::sin(phi)*std::cos(theta);
            pos[1] = radius*std::sin(phi)*std::sin(theta);
            pos[2] = radius*std::cos(phi);

            nor = pos.normalized();

            cloud->m_points[p].pos() = pos;
            cloud->m_points[p].normal() = nor;
            cloud->m_points[p].color() = col;
        }
    }

    // add poles
    cloud->m_points[size-2].pos() = Ra::Core::Vector3(0.0, 0.0, radius);
    cloud->m_points[size-2].normal() = Ra::Core::Vector3(0.0, 0.0, +1.0);
    cloud->m_points[size-2].color() = Ra::Core::Vector4(0.5, 1.0, 0.5, 1.0);

    cloud->m_points[size-1].pos() = Ra::Core::Vector3(0.0,0.0,-radius);
    cloud->m_points[size-1].normal() = Ra::Core::Vector3(0.0, 0.0, -1.0);
    cloud->m_points[size-1].color() = Ra::Core::Vector4(0.5, 0.0, 0.5, 1.0);

    return cloud;
}

std::shared_ptr<PointyCloud> PointyCloudFactory::makeRandom(int n, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
{
    std::shared_ptr<PointyCloud> cloud = std::make_shared<PointyCloud>();

    cloud->m_points.resize(n);

    Eigen::Array3f min;
    min << xmin, ymin, zmin;
    Eigen::Array3f max;
    max << xmax, ymax, zmax;

    for(int i=0; i<n; ++i)
    {
        cloud->m_points[i].pos() = (0.5*(Ra::Core::Vector3::Random()+Ra::Core::Vector3::Ones()).array()*(max-min)+min).matrix();
        cloud->m_points[i].normal() = Ra::Core::Vector3::Random().normalized();
        cloud->m_points[i].color() = 0.5*( Ra::Core::Vector4::Random() + Ra::Core::Vector4::Ones() );
    }

    return cloud;
}

}

