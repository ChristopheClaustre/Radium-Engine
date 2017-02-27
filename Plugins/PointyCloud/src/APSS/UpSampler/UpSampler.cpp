#include "UpSampler.hpp"

namespace PointyCloudPlugin
{

UpSampler::UpSampler(std::shared_ptr<PointyCloud> originalCloud) :
    m_originalCloud(originalCloud), m_cloud(new PointyCloud()), m_prevCloud(new PointyCloud()),
    m_upsamplingInfo(new std::map<unsigned int, UpsamplingInfo>()), m_prevUpsamplingInfo(new std::map<unsigned int, UpsamplingInfo>())
{
}

UpSampler::~UpSampler()
{
}

void UpSampler::upSampleCloud(const std::vector<unsigned int>& indices, int N)
{
    #pragma omp parallel for
    for ( uint i = 0 ; i < N ; ++i )
    {
        this->upSamplePointMaster(indices[i]);
    }

    std::swap(m_prevCloud,m_cloud);
    m_cloud->clear();

    std::swap(m_prevUpsamplingInfo, m_upsamplingInfo);
    m_upsamplingInfo->clear();
}

// m x m = nb de splats
void UpSampler::upSamplePoint(const int& m, const APoint& originalPoint, int index)
{
    std::map<unsigned int, UpsamplingInfo>::iterator it = m_prevUpsamplingInfo->find(index);

    if (!(originalPoint.eligible()) || m <= 1)
    {
        #pragma omp critical (m_cloud_insertion)
        {
            m_cloud->push_back(originalPoint);
        }
    }
    else
    {
        if (it != m_prevUpsamplingInfo->end() && it->second.m_M == m)
        {
            int mm = m*m;
            const UpsamplingInfo& prevM = it->second;

            int n;
            #pragma omp critical (m_cloud_insertion)
            {
                APoint temp(originalPoint.pos(),originalPoint.normal(),
                            originalPoint.color(),originalPoint.radius()/m,false);
                n =  m_cloud->size();
                m_cloud->resize(n+mm, temp);
            }

            for (int i = 0; i < mm; ++i) {
                m_cloud->at(n+i).pos() = m_prevCloud->at(prevM.m_begin+i).pos();
                m_cloud->at(n+i).normal() = m_prevCloud->at(prevM.m_begin+i).normal();
            }

            UpsamplingInfo currentM; currentM.m_begin = n; currentM.m_M = m;
            #pragma omp critical (m_cloud_insertion)
            {
                m_upsamplingInfo->insert({index, currentM});
            }
        }
        else
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
            int n;
            #pragma omp critical (m_cloud_insertion)
            {
                APoint temp(centerVertice,normal,color,newRadius);
                n =  m_cloud->size();
                m_cloud->resize(n+m*m, temp);
            }

            Scalar med = (m-1)/2.0;
            Ra::Core::Vector3 firstVertice = centerVertice + u_pas * -med + v_pas * -med;
            for ( int i = 0 ; i < m; ++i )
            {
                int nim = n+i*m;
                for ( int j = 0 ; j < m ; ++j )
                {
                     m_cloud->at(nim+j).pos() = firstVertice + i * u_pas + j * v_pas;
                }
            }

            UpsamplingInfo currentM; currentM.m_begin = n; currentM.m_M = m;
            #pragma omp critical (m_cloud_insertion)
            {
                m_upsamplingInfo->insert({index, currentM});
            }
        }
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
