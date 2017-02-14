#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <Core/RaCore.hpp>
#include <Core/Math/LinearAlgebra.hpp>
#include <Core/Containers/VectorArray.hpp>
#include <Core/Mesh/MeshTypes.hpp>
#include <Engine/Renderer/Mesh/Mesh.hpp>

namespace PointyCloudPlugin {

class PointyCloud
{
public:
    /// Create an empty mesh
    inline PointyCloud() {}

    /// constructor with parameters
    inline PointyCloud(Ra::Engine::Mesh *);

    /// Copy constructor and assignment operator
    PointyCloud( const PointyCloud& ) = default;
    PointyCloud& operator= ( const PointyCloud& ) = default;

    /// Appends another pointycloud to this one.
    inline void append( const PointyCloud& other );

    /// Load the point cloud in a Mesh
    inline void loadToMesh(Ra::Engine::Mesh *);

public:
    Ra::Core::Vector3Array m_vertices;
    Ra::Core::Vector3Array m_normals;
    Ra::Core::Vector4Array m_colors;
};

} // namespace PointyCloudPlugin

#endif // POINTCLOUD_H
