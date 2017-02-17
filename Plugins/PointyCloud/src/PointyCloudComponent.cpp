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

using Ra::Engine::ComponentMessenger;

namespace PointyCloudPlugin
{
    PointyCloudComponent::PointyCloudComponent(const std::string& name, const Ra::Engine::Camera *camera)
        : Ra::Engine::Component( name ), m_camera( camera ),
          m_originalCloud(std::make_shared<PointyCloud>()), m_upsampler(new UpSamplerUnshaken(3,1)),
          m_culling(new UsefulPointsSelection(m_originalCloud, m_camera)),
          m_selector(std::make_shared<NeighborsSelection>(m_originalCloud, 3)),
          m_projection(new OrthogonalProjection(m_selector, m_originalCloud, 3))
    {
    }

    PointyCloudComponent::~PointyCloudComponent()
    {
        delete m_culling;
        delete m_upsampler;
        delete m_projection;
        m_selector.reset();
        m_originalCloud.reset();
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

        std::string cloudName = name;
        cloudName.append( "_Cloud" );

        m_contentName = data->getName();

        m_workingCloud = Ra::Core::make_shared<Ra::Engine::Mesh>( cloudName, GL_POINTS );

        Ra::Core::Transform T = data->getFrame();
        Ra::Core::Transform N;
        N.matrix() = (T.matrix()).inverse().transpose();

        Ra::Core::Vector3Array vertices;
        Ra::Core::Vector3Array normals;

        for ( const auto& v : data->getVertices() ) vertices.push_back( T * v );
        for ( const auto& v : data->getNormals() )  normals.push_back( (N * v).normalized() );

        m_workingCloud->loadPointyGeometry( vertices, normals );

        Ra::Core::Vector4Array colors;
        if (data->hasColors()) {
            for ( const auto& v : data->getColors() ) colors.push_back( v );
        }
        else {
            LOGP(logINFO) << "cloud " << cloudName << "has no color. Creation of colors.";
            Ra::Core::Color white;
            white << 1.0, 1.0, 1.0, 1.0;
            colors.resize(vertices.size(), white);
        }

        m_workingCloud->addData( Ra::Engine::Mesh::VERTEX_COLOR, colors);

        auto config = Ra::Engine::ShaderConfigurationFactory::getConfiguration("Pointy");

        Ra::Engine::RenderObject* ro = Ra::Engine::RenderObject::createRenderObject(roName, this, Ra::Engine::RenderObjectType::Pointy, m_workingCloud, config);

        m_meshIndex = addRenderObject(ro);

        auto sys = static_cast<PointyCloudSystem*>(m_system);

        m_originalCloud->loadFromMesh(m_workingCloud.get());
        setUpsamplingMethod(sys->getUpsamplingMethod());
        setProjectionMethod(sys->getProjectionMethod());
        setOptimizationByOctree(sys->isOptimizedByOctree());
        setOptimizationByCUDA(sys->isOptimizedByCUDA());

        LOGP(logINFO) << "cloud " << cloudName << " has " << m_originalCloud->m_points.size() << " point(s).";
//        LOGP(logINFO) << "cloud " << cloudName << " is pointy at " << m_originalCloud->m_points.size() << " level(s).";

// FOR APSS TEST (comment computePointyCloud's body)
//        PointyCloud points = m_culling->selectUsefulPoints();
//        m_upsampler->upSampleCloud(&points);
//        m_projection->project(points);
//        points.loadToMesh(m_workingCloud.get());
    }

    void PointyCloudComponent::computePointyCloud()
    {
        PointyCloud points = m_culling->selectUsefulPoints();
        m_upsampler->upSampleCloud(points);
        m_projection->project(points);
        points.loadToMesh(m_workingCloud.get());
    }

    void PointyCloudComponent::setInfluenceRadius(Scalar influenceRadius) {
        m_projection->setInfluenceRadius(influenceRadius);
        m_selector->setInfluenceRadius(influenceRadius);
    }

    void PointyCloudComponent::setBeta(Scalar beta) {
        // TODO donner beta à la projection
    }

    void PointyCloudComponent::setThreshold(int threshold) {
        // TODO donner le threshold a quelqu'un
    }

    void PointyCloudComponent::setM(int M) {
        // TODO donner au ré-echantillonnage
    }

    void PointyCloudComponent::setUpsamplingMethod(UPSAMPLING_METHOD method) {
        // TODO switcher entre les méthodes
        auto sys = static_cast<PointyCloudSystem*>(m_system);
        delete m_upsampler;
        m_upsampler = new UpSamplerUnshaken(sys->getM(), sys->getInfluenceRadius());
    }

    void PointyCloudComponent::setProjectionMethod(PROJECTION_METHOD method) {
        // TODO switcher entre les méthodes
        auto sys = static_cast<PointyCloudSystem*>(m_system);
        delete m_projection;
        m_projection = new OrthogonalProjection(m_selector, m_originalCloud, sys->getInfluenceRadius());
    }

    void PointyCloudComponent::setOptimizationByOctree(bool octree) {
        // TODO switcher entre avec et sans octree
        // TODO vérifier que la valeur change bien avant de swaper (risque de recréer la regular grid pour rien)
        auto sys = static_cast<PointyCloudSystem*>(m_system);
        m_selector.reset(new NeighborsSelection(m_originalCloud, sys->getInfluenceRadius()));
        setUpsamplingMethod(sys->getUpsamplingMethod());
        setProjectionMethod(sys->getProjectionMethod());
    }

    void PointyCloudComponent::setOptimizationByCUDA(bool cuda) {
        // TODO donner à tout le monde ???
    }

} // namespace PointyCloudPlugin
