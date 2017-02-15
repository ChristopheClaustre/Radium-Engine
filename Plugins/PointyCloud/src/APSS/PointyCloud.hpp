#ifndef POINTYCLOUD_POINTYCLOUD_H
#define POINTYCLOUD_POINTYCLOUD_H

#include <Core/RaCore.hpp>
#include <Core/Math/LinearAlgebra.hpp>
#include <Core/Containers/VectorArray.hpp>
#include <Core/Mesh/MeshTypes.hpp>
#include <Engine/Renderer/Mesh/Mesh.hpp>

namespace PointyCloudPlugin {

class APoint
{
public:
    // required by Patate
    typedef float Scalar; //TODO use Radium's Scalar, in case 'double' would be used
    typedef Ra::Core::Vector3 VectorType;

    inline APoint(  const Ra::Core::Vector3& _pos =    Ra::Core::Vector3::Zero(),
                    const Ra::Core::Vector3& _normal = Ra::Core::Vector3::Zero(),
                    const Ra::Core::Vector4& _color =  Ra::Core::Vector3::Zero()
                    )
        : m_pos(_pos), m_normal(_normal), m_color(_color) {}

    inline const Ra::Core::Vector3& pos()    const { return m_pos; }
    inline const Ra::Core::Vector3& normal() const { return m_normal; }
    inline const Ra::Core::Vector4& color() const  { return m_color; }

    inline Ra::Core::Vector3& pos()    { return m_pos; }
    inline Ra::Core::Vector3& normal() { return m_normal; }
    inline Ra::Core::Vector4& color()  { return m_color; }

private:
    Ra::Core::Vector3 m_pos, m_normal;
    Ra::Core::Vector4 m_color;
};

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
    std::vector<APoint> m_points;
};

} // namespace PointyCloudPlugin

#include <APSS/PointyCloud.inl>

#endif // POINTYCLOUD_POINTYCLOUD_H
