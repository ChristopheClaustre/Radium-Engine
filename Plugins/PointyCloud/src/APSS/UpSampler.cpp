#include "UpSampler.hpp"

namespace PointyCloudPlugin
{

UpSampler::UpSampler(Scalar radius) : m_radius(radius), m_cloud( nullptr)
{
}

UpSampler::~UpSampler()
{
}

// m x m = nb de splats
void UpSampler::upSamplePoint(const int &m, const int& indice )
{
    APoint centerPoint = m_cloud->m_points[indice];
    if (centerPoint.isEligible())
    {
        const Ra::Core::Vector3 &normal = centerPoint.normal();
        const Ra::Core::Vector3 &u = this->calculU(normal);
        const Ra::Core::Vector3 &v = this->calculV(normal, u);

        const Ra::Core::Vector3 &u_pas =  u * m_radius * 2  / (m-1);
        const Ra::Core::Vector3 &v_pas =  v * m_radius * 2  / (m-1);
        const Ra::Core::Vector3 &centerVertice = centerPoint.pos();

        const Ra::Core::Vector4 &color = m_cloud->m_points[indice].color();

        const Ra::Core::Vector3 &topLeftVertice = Ra::Core::Vector3(u * -m_radius+v * m_radius) + centerVertice;
        for (int i = 0 ; i < m ; i ++ )
        {
            for (int j = 0 ; j < m ; j ++ )
            {
                APoint newPoint(Ra::Core::Vector3( i * u_pas + j * -v_pas)+topLeftVertice ,normal,color);
                m_newpoints.push_back(newPoint);
            }
        }
    }
    else
    {
        m_newpoints.push_back(centerPoint);
    }
}

// (0,0,0) appartient à tous les plans donc il suffit d'avoir 1 pt A du plan pour avoir le vecteur directeur AO
Ra::Core::Vector3 UpSampler::calculU(const Ra::Core::Vector3& normal)
{
    Ra::Core::Vector3 u;
     if ( normal[2] != 0)
        u = Ra::Core::Vector3(1,1,(normal[0]+normal[1]) / ( -normal[2] ) );
    else if ( normal[1] != 0)
        u = Ra::Core::Vector3(1,(normal[0]+normal[2]) / ( -normal[1] ),1 );
    else if ( normal[0] != 0)
        u = Ra::Core::Vector3((normal[1]+normal[2]) / ( -normal[0] ),1,1 );
    else
    {
        std::cerr << "Error normal = " << normal<< std::endl;
        return Ra::Core::Vector3(0,0,0);
    }
    u.normalize();
    return u;
}
// crossProd de 2 vect donne un vect ortho aux 2 ;
Ra::Core::Vector3 UpSampler::calculV(const Ra::Core::Vector3& normal, const Ra::Core::Vector3& u)
{
    Ra::Core::Vector3 v = u.cross(normal);
    v.normalize();
    return v;
}

}
