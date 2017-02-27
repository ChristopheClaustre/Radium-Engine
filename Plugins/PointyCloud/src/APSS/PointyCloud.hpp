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
                    const Ra::Core::Vector4& _color =   Ra::Core::Vector4::Zero(),
                    const Scalar& _radius =          0.,
                    const bool& _eligible =             true
                    )
        : m_pos(_pos), m_normal(_normal), m_color(_color), m_radius(_radius), m_eligible(_eligible) {}

    inline const VectorType& pos()          const { return m_pos; }
    inline const VectorType& normal()       const { return m_normal; }
    inline const Ra::Core::Vector4& color() const { return m_color; }
    inline const Scalar& radius()           const { return m_radius; }
    inline const bool& eligible()           const { return m_eligible; }

    inline VectorType& pos()            { return m_pos; }
    inline VectorType& normal()         { return m_normal; }
    inline Ra::Core::Vector4& color()   { return m_color; }
    inline Scalar& radius()             { return m_radius; }
    inline bool& eligible()             { return m_eligible; }

private:
    VectorType m_pos;
    VectorType m_normal;
    Ra::Core::Vector4 m_color;
    Scalar m_radius;
    bool m_eligible;
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

    /// getter on m_points
    inline Ra::Core::AlignedStdVector<APoint>& points() { return m_points; }
    inline const Ra::Core::AlignedStdVector<APoint>& points() const { return m_points; }

    /// Appends another pointycloud to this one.
    inline void append( const PointyCloud& other );

    /// Load the point cloud from a Mesh
    inline void loadFromMesh(const Ra::Engine::Mesh *);

    /// Load the point cloud into a Mesh
    inline void loadToMesh(Ra::Engine::Mesh *) const;

    /// subscript operator (shortcut for m_point's subscript operator)
    inline APoint& operator[](const int& i) { return m_points[i]; }
    inline APoint& at(const int& i)  { return m_points[i]; }
    inline const APoint& operator[](const int& i) const  { return m_points[i]; }
    inline const APoint& at(const int& i) const  { return m_points[i]; }

    /// shortcut for size method on m_points
    inline size_t size() { return m_points.size(); }

    /// shortcut for push_back on m_points
    inline void push_back(APoint p);

    /// shortcut for resize method on m_points
    inline void resize(int n);
    inline void resize(int n, APoint init);

    /// shortcut for clear method on m_points
    inline void clear();

    /// swap two APoint
    inline void swap(int indice1, int indice2);

public:
    /// vector of APoints
    Ra::Core::AlignedStdVector<APoint> m_points;
}; // class PointyCloud



} // namespace PointyCloudPlugin

#include <APSS/PointyCloud.inl>

#endif // POINTYCLOUD_POINTYCLOUD_H
