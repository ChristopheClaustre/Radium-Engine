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

#include <APSS/UsefulPointsSelection.hpp>
#include <APSS/PointyCloud.hpp>
#include <APSS/OrthogonalProjection.hpp>
#include <APSS/NeighborsSelection.hpp>
#include <APSS/UpSamplerUnshaken.hpp>
#include <APSS/UpSamplerSimple.hpp>

using Ra::Engine::ComponentMessenger;

namespace PointyCloudPlugin
{
    PointyCloudComponent::PointyCloudComponent(const std::string& name, const Ra::Engine::Camera *camera)
        : Ra::Engine::Component( name ), m_camera( camera )
    {
    }

    PointyCloudComponent::~PointyCloudComponent()
    {
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

        m_workingCloud = Ra::Core::make_shared<Ra::Engine::Mesh>( meshName, GL_POINTS );

        Ra::Core::Transform T = data->getFrame();
        Ra::Core::Transform N;
        N.matrix() = (T.matrix()).inverse().transpose();

        Ra::Core::Vector3Array vertices;
        Ra::Core::Vector3Array normals;

        for ( const auto& v : data->getVertices() ) vertices.push_back( T * v );
        for ( const auto& v : data->getNormals() )  normals.push_back( (N * v).normalized() );

        m_workingCloud->loadPointyGeometry( vertices, normals );

        Ra::Core::Vector4Array colors;

        for ( const auto& v : data->getColors() )     colors.push_back( v );

        // colors required
        if(colors.size()!=vertices.size())
        {
            Ra::Core::Color white;
            white << 1.0, 1.0, 1.0, 1.0;
            colors.resize(vertices.size(), white);
        }

        m_workingCloud->addData( Ra::Engine::Mesh::VERTEX_COLOR,     colors     );

        auto config = Ra::Engine::ShaderConfigurationFactory::getConfiguration("Pointy");

        Ra::Engine::RenderObject* ro = Ra::Engine::RenderObject::createRenderObject(roName, this, Ra::Engine::RenderObjectType::Pointy, m_workingCloud, config);

        m_meshIndex = addRenderObject(ro);

        m_originalCloud = std::make_shared<PointyCloud>(m_workingCloud.get());
        m_selector = std::make_shared<NeighborsSelection>(m_originalCloud, 1.0);
        m_culling = new UsefulPointsSelection(m_originalCloud, m_camera);

        //TODO (xavier) Passer l'attribut rayon du component
//        m_upsampler = new UpSamplerUnshaken(1,3);
        m_upsampler = new UpSamplerSimple(2.0f,1.0f,*m_camera);
        m_projection = new OrthogonalProjection(m_selector, m_originalCloud, 1.0);
    }

    void PointyCloudComponent::computePointyCloud()
    {
        PointyCloud points = m_culling->selectUsefulPoints();
//        m_projection->project(points);
        m_upsampler->upSampleCloud(points);
        points.loadToMesh(m_workingCloud.get());
    }

    void PointyCloudComponent::setInfluenceRadius(float influenceRadius) {
        // TODO donner l'influence Radius à la selection des voisins
        m_projection->setInfluenceRadius(influenceRadius);
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
