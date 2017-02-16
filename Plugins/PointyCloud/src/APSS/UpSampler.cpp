#include "UpSampler.hpp"


namespace PointyCloudPlugin
{


UpSampler::UpSampler(float rayon) : m_rayon(rayon), m_cloud( nullptr){
}

UpSampler::~UpSampler(){

}

void UpSampler::upSampleCloud(PointyCloud cloud){

    m_cloud = cloud;
    m_points.clear();
    const int &n = m_cloud->m_points.size() ;

    for ( uint i = 0 ; i < n ; i++ )
    {
        this->upSamplePoint( getM(i), i);
    }
    m_cloud->m_points = m_points;
}


// m x m = nb de splats
void UpSampler::upSamplePoint(const int &m, const int& indice ){

    const Ra::Core::Vector3 &normal = m_cloud->normal()[indice];
    const Ra::Core::Vector3 &u = this->calculU(normal);
    const Ra::Core::Vector3 &v = this->calculV(normal, u);

    const Ra::Core::Vector3 &u_rayon =  u * m_rayon  / m;
    const Ra::Core::Vector3 &v_rayon =  v * m_rayon  / m;
    const Ra::Core::Vector3 &centerVertice = m_cloud->pos()[indice];

    const Ra::Core::Vector4 &color = m_cloud->color()[indice];

    if ( m % 2 == 1 )
    {
        for (int i = -(m -1 )  / 2 ; i <= (m-1) /2 ; i ++ )
        {
            for (int j = - (m -1 ) / 2 ; j <= (m-1) / 2 ; j ++ )
            {
                APoint newPoint(Ra::Core::Vector3( 2  * i * u_rayon + 2 * j * v_rayon ) + centerVertice,normal,color);
                m_points.push_back(newPoint);
//                m_cloud->addData(Ra::Engine::Mesh::Vec4Data::VERTEX_COLOR, m_cloud->getData(Ra::Engine::Mesh::Vec4Data::VERTEX_COLOR));
            }
        }
    }
    else
    {
        const Ra::Core::Vector3 &topLeftVertice = (m /2 - 1) * 2 * (-u_rayon) + (m /2 - 1) * 2 * v_rayon + - u_rayon + v_rayon + centerVertice;
        for (int i = 0 ; i < m ; i ++ )
        {
            for (int j = 0 ; j < m ; j ++ )
            {
                APoint newPoint(Ra::Core::Vector3( 2  * i * u_rayon + 2 * j * -v_rayon + topLeftVertice ),normal,color);
                m_points.push_back(newPoint);
//                m_cloud->addData(Ra::Engine::Mesh::Vec4Data::VERTEX_COLOR, m_cloud->getData(Ra::Engine::Mesh::Vec4Data::VERTEX_COLOR));
            }
        }
    }
    // TODO actualiser le rayon

}

int UpSampler::getM(int indice)
{
    return (3);
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
