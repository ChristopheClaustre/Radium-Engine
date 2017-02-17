#ifndef POINTYCLOUDPLUGIN_POINTYCLOUD_HPP
#define POINTYCLOUDPLUGIN_POINTYCLOUD_HPP

#include <Core/RaCore.hpp>
#include <Core/Math/LinearAlgebra.hpp>
#include <Core/Containers/VectorArray.hpp>
#include <Core/Mesh/MeshTypes.hpp>
#include <Engine/Renderer/Mesh/Mesh.hpp>

namespace PointyCloudPlugin {

namespace ForPatate {
    typedef Scalar _Scalar;
    typedef Ra::Core::Vector3 _VectorType;
}

class APoint
{
public:
    // required by Patate
    typedef ForPatate::_Scalar Scalar;
    typedef ForPatate::_VectorType VectorType;

    inline APoint(  const VectorType& _pos =            VectorType::Zero(),
                    const VectorType& _normal =         VectorType::Zero(),
                    const Ra::Core::Vector4& _color =   VectorType::Zero()
                    )
        : m_pos(_pos), m_normal(_normal), m_color(_color) {}

    inline const VectorType& pos()    const { return m_pos; }
    inline const VectorType& normal() const { return m_normal; }
    inline const Ra::Core::Vector4& color()  const  { return m_color; }

    inline VectorType& pos()    { return m_pos; }
    inline VectorType& normal() { return m_normal; }
    inline Ra::Core::Vector4& color()  { return m_color; }

private:
    VectorType m_pos, m_normal;
    Ra::Core::Vector4 m_color;
};

class PointyCloud
{
public:
    /// Create an empty mesh
    inline PointyCloud() {}

    /// constructor with parameters
    inline PointyCloud(const Ra::Engine::Mesh *);

    /// Copy constructor and assignment operator
    PointyCloud( const PointyCloud& ) = default;
    PointyCloud& operator= ( const PointyCloud& ) = default;

    /// Appends another pointycloud to this one.
    inline void append( const PointyCloud& other );

    /// Load the point cloud from a Mesh
    inline void loadFromMesh(const Ra::Engine::Mesh *);

    /// Load the point cloud into a Mesh
    inline void loadToMesh(Ra::Engine::Mesh *);

public:
    std::vector<APoint> m_points;
};

} // namespace PointyCloudPlugin

#include <APSS/PointyCloud.inl>

#endif // POINTYCLOUD_POINTYCLOUD_H
