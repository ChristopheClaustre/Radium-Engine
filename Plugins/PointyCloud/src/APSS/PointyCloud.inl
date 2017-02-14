#include "PointyCloud.hpp"

namespace PointyCloudPlugin {

inline PointyCloud::PointyCloud(Ra::Engine::Mesh * mesh)
{
    Ra::Core::TriangleMesh geometry = mesh->getGeometry();
    Ra::Core::Vector4Array colors = mesh->getData(Ra::Engine::Mesh::VERTEX_COLOR);

    m_vertices.insert( m_vertices.end(), geometry.m_vertices.cbegin(), geometry.m_vertices.cend() );
    m_normals.insert( m_normals.end(), geometry.m_normals.cbegin(), geometry.m_normals.cend() );
    m_colors.insert( m_colors.end(), colors.cbegin(), colors.cend() );
}

inline PointyCloud::loadToMesh(Ra::Engine::Mesh * mesh) {
    mesh->loadPointyGeometry(m_vertices, m_normals);
    mesh->addData(Ra::Engine::Mesh::VERTEX_COLOR, m_colors);
}

inline void PointyCloud::append( const PointyCloud& other )
{
    m_vertices.insert( m_vertices.end(), other.m_vertices.cbegin(), other.m_vertices.cend() );
    m_normals.insert( m_normals.end(), other.m_normals.cbegin(), other.m_normals.cend() );
    m_colors.insert( m_colors.end(), other.m_colors.cbegin(), other.m_colors.cend() );
}

} // namespace PointyCloudPlugin
