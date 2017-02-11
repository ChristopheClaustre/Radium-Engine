#include <PointyCloudComponent.hpp>

#include <APSS/UsefulPointsSelection.hpp>

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

using Ra::Engine::ComponentMessenger;

namespace PointyCloudPlugin
{
    PointyCloudComponent::PointyCloudComponent(const std::string& name, const Ra::Engine::Camera *camera)
        : Ra::Engine::Component( name ), m_camera( camera )
    {
    }

    PointyCloudComponent::~PointyCloudComponent()
    {
        delete m_originalCloud;
        delete m_culling;
    }

    void PointyCloudComponent::initialize()
    {
    }

    void PointyCloudComponent::handlePointyCloudLoading( const Ra::Asset::GeometryData* data )
    {
        std::string name( m_name );
        name.append( "_" + data->getName() );

        std::string roName = name;
        roName.append( "_RO" );

        std::string meshName = name;
        meshName.append( "_Mesh" );

        m_contentName = data->getName();

        m_originalCloud = new Ra::Engine::Mesh( meshName, GL_POINTS );

        Ra::Core::Transform T = data->getFrame();
        Ra::Core::Transform N;
        N.matrix() = (T.matrix()).inverse().transpose();

        Ra::Core::Vector3Array vertices;
        Ra::Core::Vector3Array normals;

        for ( const auto& v : data->getVertices() ) vertices.push_back( T * v );
        for ( const auto& v : data->getNormals() )  normals.push_back( (N * v).normalized() );

        m_originalCloud->loadPointyGeometry( vertices, normals );

        Ra::Core::Vector3Array tangents;
        Ra::Core::Vector3Array bitangents;
        Ra::Core::Vector3Array texcoords;
        Ra::Core::Vector4Array colors;

        for ( const auto& v : data->getTangents() )   tangents.push_back( v );
        for ( const auto& v : data->getBiTangents() ) bitangents.push_back( v );
        for ( const auto& v : data->getTexCoords() )  texcoords.push_back( v );
        for ( const auto& v : data->getColors() )     colors.push_back( v );

        m_originalCloud->addData( Ra::Engine::Mesh::VERTEX_TANGENT,   tangents   );
        m_originalCloud->addData( Ra::Engine::Mesh::VERTEX_BITANGENT, bitangents );
        m_originalCloud->addData( Ra::Engine::Mesh::VERTEX_TEXCOORD,  texcoords  );
        m_originalCloud->addData( Ra::Engine::Mesh::VERTEX_COLOR,     colors     );

        auto config = Ra::Engine::ShaderConfigurationFactory::getConfiguration("Pointy");

        m_workingCloud = Ra::Core::make_shared<Ra::Engine::Mesh>( meshName, GL_POINTS );
        resetWorkingCloud();

        Ra::Engine::RenderObject* ro = Ra::Engine::RenderObject::createRenderObject(roName, this, Ra::Engine::RenderObjectType::Pointy, m_workingCloud, config);

        m_meshIndex = addRenderObject(ro);

        m_culling = new UsefulPointsSelection(m_originalCloud, m_camera);
    }

    void PointyCloudComponent::resetWorkingCloud() {
        m_workingCloud->loadPointyGeometry(
                                m_originalCloud->getGeometry().m_vertices,
                                m_originalCloud->getGeometry().m_normals
                    );

        m_workingCloud->addData( Ra::Engine::Mesh::VERTEX_TANGENT,   m_originalCloud->getData(Ra::Engine::Mesh::VERTEX_TANGENT));
        m_workingCloud->addData( Ra::Engine::Mesh::VERTEX_BITANGENT, m_originalCloud->getData(Ra::Engine::Mesh::VERTEX_BITANGENT));
        m_workingCloud->addData( Ra::Engine::Mesh::VERTEX_TEXCOORD,  m_originalCloud->getData(Ra::Engine::Mesh::VERTEX_TEXCOORD));
        m_workingCloud->addData( Ra::Engine::Mesh::VERTEX_COLOR,     m_originalCloud->getData(Ra::Engine::Mesh::VERTEX_COLOR));
    }

    void PointyCloudComponent::computePointyCloud()
    {
        //TODO: l'APSS :p
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

    void PointyCloudComponent::setInfluenceRadius(float influenceRadius) {
        // TODO donner l'influence Radius à la selection des voisins
    }

    void PointyCloudComponent::setBeta(float beta) {
        // TODO donner beta à la projection
    }

    void PointyCloudComponent::setThreshold(int threshold) {
        // TODO donner le treshold au ré-echantillonage il me semble
    }

    void PointyCloudComponent::setM(int M) {
        // TODO donner au ré-echantillonnage
    }

    void PointyCloudComponent::setUpsamplingMethod(UPSAMPLING_METHOD method) {
        // TODO donner au ré-echantillonnage
    }

    void PointyCloudComponent::setProjectionMethod(PROJECTION_METHOD method) {
        // TODO donner à la projection
    }

    void PointyCloudComponent::setOptimizationByOctree(bool octree) {
        // TODO donner à la selection des voisins
    }

    void PointyCloudComponent::setOptimizationByCUDA(bool cuda) {
        // TODO donner à tout le monde ???
    }

} // namespace PointyCloudPlugin
