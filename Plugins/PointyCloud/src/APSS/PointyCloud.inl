#include "PointyCloud.hpp"

namespace PointyCloudPlugin {

inline PointyCloud::PointyCloud(Ra::Engine::Mesh * mesh)
{
    Ra::Core::TriangleMesh geometry = mesh->getGeometry();
    Ra::Core::Vector4Array colors = mesh->getData(Ra::Engine::Mesh::VERTEX_COLOR);

    for (int i = 0; i < geometry.m_vertices.size(); ++i)
    {
        m_points.push_back(APoint(geometry.m_vertices[i], geometry.m_normals[i], colors[i]));
    }
}

inline void PointyCloud::loadToMesh(Ra::Engine::Mesh * mesh) {
    Ra::Core::Vector3Array vertices;
    Ra::Core::Vector3Array normals;
    Ra::Core::Vector4Array colors;

    for (int i = 0; i < m_points.size(); ++i)
    {
        APoint point = m_points[i];

        vertices.push_back(point.pos());
        normals.push_back(point.normal());
        colors.push_back(point.color());
    }

    mesh->loadPointyGeometry(vertices, normals);
    mesh->addData(Ra::Engine::Mesh::VERTEX_COLOR, colors);
}

inline void PointyCloud::append( const PointyCloud& other )
{
    m_points.insert( m_points.end(), other.m_points.cbegin(), other.m_points.cend() );
}

} // namespace PointyCloudPlugin
