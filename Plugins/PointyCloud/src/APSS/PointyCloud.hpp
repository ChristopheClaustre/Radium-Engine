#ifndef POINTYCLOUDPLUGIN_POINTYCLOUD_HPP
#define POINTYCLOUDPLUGIN_POINTYCLOUD_HPP

#include "Point.h"

namespace Ra {
namespace Engine {
    class Mesh;
} // namespace Engine
} // namespace Ra

namespace PointyCloudPlugin {

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
