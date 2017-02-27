#include "PointyCloud.hpp"

namespace PointyCloudPlugin {

inline PointyCloud::PointyCloud(const Ra::Engine::Mesh * mesh)
{
    loadFromMesh(mesh);
}

inline void PointyCloud::loadFromMesh(const Ra::Engine::Mesh * mesh) {
    Ra::Core::TriangleMesh geometry = mesh->getGeometry();
    Ra::Core::Vector4Array colors = mesh->getData(Ra::Engine::Mesh::VERTEX_COLOR);
    Ra::Core::Vector1Array radiuses = mesh->getData(Ra::Engine::Mesh::POINT_RADIUS);

    m_points.clear();
    for (int i = 0; i < geometry.m_vertices.size(); ++i)
    {
        m_points.push_back(APoint(geometry.m_vertices[i], geometry.m_normals[i], colors[i], radiuses[i]));
    }
}

inline void PointyCloud::loadToMesh(Ra::Engine::Mesh * mesh) const {

    size_t size = m_points.size();

    Ra::Core::Vector3Array vertices(size);
    Ra::Core::Vector3Array normals(size);
    Ra::Core::Vector4Array colors(size);
    Ra::Core::Vector1Array radiuses(size);

    for (int i = 0; i < size; ++i)
    {
        vertices[i] = m_points[i].pos();
        normals[i] =  m_points[i].normal();
        colors[i] =   m_points[i].color();
        radiuses[i] = m_points[i].radius();
    }

    mesh->loadPointyGeometry(vertices, normals);
    mesh->addData(Ra::Engine::Mesh::VERTEX_COLOR, colors);
    mesh->addData(Ra::Engine::Mesh::POINT_RADIUS, radiuses);
}

inline void PointyCloud::append( const PointyCloud& other )
{
    m_points.insert( m_points.end(), other.m_points.cbegin(), other.m_points.cend() );
}

inline void PointyCloud::push_back(APoint p) {
    m_points.push_back(p);
}

inline void PointyCloud::resize(int n) {
    m_points.resize(n);
}

inline void PointyCloud::resize(int n, APoint init) {
    m_points.resize(n, init);
}

inline void PointyCloud::clear() {
    m_points.clear();
}

inline void PointyCloud::swap(int indice1, int indice2) {
    std::swap(m_points[indice1], m_points[indice2]);
}

} // namespace PointyCloudPlugin
