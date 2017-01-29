#include <PointyCloudComponent.hpp>

#include <iostream>

#include <Core/String/StringUtils.hpp>
#include <Core/Mesh/MeshUtils.hpp>

#include <Core/Geometry/Normal/Normal.hpp>

#include <Engine/Renderer/RenderObject/RenderObjectManager.hpp>
#include <Engine/Managers/ComponentMessenger/ComponentMessenger.hpp>

#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <Engine/Renderer/RenderObject/RenderObject.hpp>
#include <Engine/Renderer/RenderObject/RenderObjectTypes.hpp>
#include <Engine/Renderer/RenderTechnique/RenderTechnique.hpp>
#include <Engine/Renderer/RenderTechnique/ShaderProgram.hpp>
#include <Engine/Renderer/RenderTechnique/ShaderProgramManager.hpp>
#include <Engine/Renderer/RenderObject/Primitives/DrawPrimitives.hpp>
#include <Engine/Renderer/RenderTechnique/ShaderConfigFactory.hpp>

#include <Engine/Assets/FileData.hpp>
#include <Engine/Assets/GeometryData.hpp>

using Ra::Core::TriangleMesh;
using Ra::Engine::ComponentMessenger;

typedef Ra::Core::VectorArray<Ra::Core::Triangle> TriangleArray;

namespace PointyCloudPlugin
{
    PointyCloudComponent::PointyCloudComponent(const std::string& name)
        : Ra::Engine::Component( name )
    {
    }

    PointyCloudComponent::~PointyCloudComponent()
    {
    }

    void PointyCloudComponent::initialize()
    {
    }

    void PointyCloudComponent::handleMeshLoading( const Ra::Asset::GeometryData* data )
    {
        std::string name( m_name );
        name.append( "_" + data->getName() );

        std::string roName = name;
        roName.append( "_RO" );
        
#if 1
        std::string meshName = name;
        meshName.append( "_Mesh" );

        std::string matName = name;
        matName.append( "_Mat" );

        m_contentName = data->getName();

        std::shared_ptr<Ra::Engine::Mesh> displayMesh( new Ra::Engine::Mesh( meshName ) );

        Ra::Core::TriangleMesh mesh;
        Ra::Core::Transform T = data->getFrame();
        Ra::Core::Transform N;
        N.matrix() = (T.matrix()).inverse().transpose();

        for (size_t i = 0; i < data->getVerticesSize(); ++i)
        {
            mesh.m_vertices.push_back(T * data->getVertices()[i]);
            mesh.m_normals.push_back((N * data->getNormals()[i]).normalized());
        }

        for (const auto& face : data->getFaces())
        {
            mesh.m_triangles.push_back(face.head<3>());
        }

        displayMesh->loadGeometry(mesh);

        Ra::Core::Vector3Array tangents;
        Ra::Core::Vector3Array bitangents;
        Ra::Core::Vector3Array texcoords;

        Ra::Core::Vector4Array colors;

        for ( const auto& v : data->getTangents() )     tangents.push_back( v );
        for ( const auto& v : data->getBiTangents() )   bitangents.push_back( v );
        for ( const auto& v : data->getTexCoords() )    texcoords.push_back( v );
        for ( const auto& v : data->getColors() )       colors.push_back( v );

        displayMesh->addData( Ra::Engine::Mesh::VERTEX_TANGENT, tangents );
        displayMesh->addData( Ra::Engine::Mesh::VERTEX_BITANGENT, bitangents );
        displayMesh->addData( Ra::Engine::Mesh::VERTEX_TEXCOORD, texcoords );
        displayMesh->addData( Ra::Engine::Mesh::VERTEX_COLOR, colors );

        // FIXME(Charly): Should not weights be part of the geometry ?
        //        mesh->addData( Ra::Engine::Mesh::VERTEX_WEIGHTS, meshData.weights );

        Ra::Engine::Material* mat = new Ra::Engine::Material( matName );
        auto m = data->getMaterial();
        if ( m.hasDiffuse() )   mat->m_kd    = m.m_diffuse;
        if ( m.hasSpecular() )  mat->m_ks    = m.m_specular;
        if ( m.hasShininess() ) mat->m_ns    = m.m_shininess;
        if ( m.hasOpacity() )   mat->m_alpha = m.m_opacity;

#ifdef LOAD_TEXTURES
        if ( m.hasDiffuseTexture() ) mat->addTexture( Ra::Engine::Material::TextureType::TEX_DIFFUSE, m.m_texDiffuse );
        if ( m.hasSpecularTexture() ) mat->addTexture( Ra::Engine::Material::TextureType::TEX_SPECULAR, m.m_texSpecular );
        if ( m.hasShininessTexture() ) mat->addTexture( Ra::Engine::Material::TextureType::TEX_SHININESS, m.m_texShininess );
        if ( m.hasOpacityTexture() ) mat->addTexture( Ra::Engine::Material::TextureType::TEX_ALPHA, m.m_texOpacity );
        if ( m.hasNormalTexture() ) mat->addTexture( Ra::Engine::Material::TextureType::TEX_NORMAL, m.m_texNormal );
#endif

        auto config = Ra::Engine::ShaderConfigurationFactory::getConfiguration("BlinnPhong");

        Ra::Engine::RenderObject* ro = Ra::Engine::RenderObject::createRenderObject(roName, this, Ra::Engine::RenderObjectType::Fancy, displayMesh, config, mat);
        if ( mat->m_alpha < 1.0 ) ro->setTransparent(true);
#else
        auto ro = Ra::Engine::RenderObject::createFancyFromAsset(roName, this, data, true);
#endif

        m_meshIndex = addRenderObject(ro);
    }

    Ra::Core::Index PointyCloudComponent::getRenderObjectIndex() const
    {
        return m_meshIndex;
    }

    const Ra::Engine::Mesh& PointyCloudComponent::getDisplayMesh() const
    {
        return *(getRoMgr()->getRenderObject(getRenderObjectIndex())->getMesh());
    }

    Ra::Engine::Mesh& PointyCloudComponent::getDisplayMesh()
    {
        return *(getRoMgr()->getRenderObject(getRenderObjectIndex())->getMesh());
    }

    void PointyCloudComponent::setMeshInput(const std::shared_ptr<Ra::Engine::Mesh> meshShared)
    {
        getRoMgr()->getRenderObject(getRenderObjectIndex())->setMesh(meshShared);
    }

    const Ra::Core::Index* PointyCloudComponent::roIndexRead() const
    {
        return &m_meshIndex;
    }

} // namespace PointyCloudPlugin
