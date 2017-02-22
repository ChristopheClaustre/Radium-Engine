#include "PointyCloud.hpp"

namespace PointyCloudPlugin {

inline PointyCloud::PointyCloud(const Ra::Engine::Mesh * mesh)
{
    loadFromMesh(mesh);
}

inline void PointyCloud::loadFromMesh(const Ra::Engine::Mesh * mesh) {
    Ra::Core::TriangleMesh geometry = mesh->getGeometry();
    Ra::Core::Vector4Array colors = mesh->getData(Ra::Engine::Mesh::VERTEX_COLOR);
    Ra::Core::Vector1Array splatSizes = mesh->getData(Ra::Engine::Mesh::POINT_SPLATSIZE);

    m_points.clear();
    for (int i = 0; i < geometry.m_vertices.size(); ++i)
    {
        m_points.push_back(APoint(geometry.m_vertices[i], geometry.m_normals[i], colors[i], splatSizes[i]));
    }
}

inline void PointyCloud::loadToMesh(Ra::Engine::Mesh * mesh) {

    size_t size = m_points.size();

    Ra::Core::Vector3Array vertices(size);
    Ra::Core::Vector3Array normals(size);
    Ra::Core::Vector4Array colors(size);
    Ra::Core::Vector1Array splatSizes(size);

    for (int i = 0; i < size; ++i)
    {
        vertices[i] = m_points[i].pos();
        normals[i] = m_points[i].normal();
        colors[i] = m_points[i].color();
        splatSizes[i] = m_points[i].splatSize();
    }

    mesh->loadPointyGeometry(vertices, normals);
    mesh->addData(Ra::Engine::Mesh::VERTEX_COLOR, colors);
    mesh->addData(Ra::Engine::Mesh::POINT_SPLATSIZE, splatSizes);
}

inline void PointyCloud::append( const PointyCloud& other )
{
    m_points.insert( m_points.end(), other.m_points.cbegin(), other.m_points.cend() );
}

} // namespace PointyCloudPlugin
