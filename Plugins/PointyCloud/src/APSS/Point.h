#ifndef POINTYCLOUDPLUGIN_POINT_H
#define POINTYCLOUDPLUGIN_POINT_H

#include "defines.h"

#include <Core/Math/LinearAlgebra.hpp>


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

    MULTIARCH inline APoint(  const VectorType& _pos =  VectorType::Zero(),
                    const VectorType& _normal =         VectorType::Zero(),
                    const Ra::Core::Vector4& _color =   Ra::Core::Vector4::Zero(),
                    const Scalar& _splatSize =          0.,
                    const bool& _eligible =             true
                    )
        : m_pos(_pos), m_normal(_normal), m_color(_color), m_splatSize(_splatSize), m_eligible(_eligible) {}

    MULTIARCH inline const VectorType& pos()          const { return m_pos; }
    MULTIARCH inline const VectorType& normal()       const { return m_normal; }
    MULTIARCH inline const Ra::Core::Vector4& color() const { return m_color; }
    MULTIARCH inline const Scalar& splatSize()        const { return m_splatSize; }
    MULTIARCH inline const bool& eligible()           const { return m_eligible; }

    MULTIARCH inline VectorType& pos()            { return m_pos; }
    MULTIARCH inline VectorType& normal()         { return m_normal; }
    MULTIARCH inline Ra::Core::Vector4& color()   { return m_color; }
    MULTIARCH inline Scalar& splatSize()          { return m_splatSize; }
    MULTIARCH inline bool& eligible()             { return m_eligible; }

private:
    VectorType m_pos, m_normal;
    Ra::Core::Vector4 m_color;
    Scalar m_splatSize;
    bool m_eligible;
};

} // namespace PointyCloudPlugin


#endif // POINTYCLOUDPLUGIN_POINT_H
