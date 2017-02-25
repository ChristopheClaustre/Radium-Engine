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
void UpSampler::upSamplePoint(const int &m, const int& indice)
{
    APoint originalPoint = m_cloud->m_points[indice];
    if (originalPoint.eligible() && m > 1)
    {
        const Ra::Core::Vector3 &centerVertice = originalPoint.pos();
        const Ra::Core::Vector3 &normal = originalPoint.normal();
        const Ra::Core::Vector4 &color = originalPoint.color();
        const Scalar newRadius = originalPoint.radius()/m;

        const Ra::Core::Vector3 &u = this->calculU(normal);
        const Ra::Core::Vector3 &v = this->calculV(normal, u);

        const Ra::Core::Vector3 &u_pas =  u * newRadius * 2;
        const Ra::Core::Vector3 &v_pas =  v * newRadius * 2;

        // we are going to add mxm points
        APoint temp(centerVertice,normal,color,newRadius);
        int n = m_newpoints.size();
        m_newpoints.resize(n+m*m, temp);

        Scalar med = (m+1)/2.0;
        for ( int i = 0 ; i < m; ++i )
        {
            int nim = n+i*m;
            for ( int j = 0 ; j < m ; ++j )
            {
                m_newpoints[nim+j].pos() = Ra::Core::Vector3( (i-med) * u_pas + (j-med) * v_pas) + centerVertice;
            }
        }
    }
    else
    {
        m_newpoints.push_back(originalPoint);
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
        LOGP(logERROR) << "Error normal = " << normal;
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

} // namespace PointyCloudPlugin
