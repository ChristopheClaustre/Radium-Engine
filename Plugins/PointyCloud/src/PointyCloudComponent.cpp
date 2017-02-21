#include <PointyCloudComponent.hpp>

#include <Core/Containers/MakeShared.hpp>

#include <Engine/Assets/GeometryData.hpp>

#include <APSS/PointyCloud.hpp>
#include <APSS/NeighborsSelection/NeighborsSelection.hpp>
#include <APSS/NeighborsSelection/NeighborsSelectionWithRegularGrid.hpp>
#include <APSS/NeighborsSelection/RegularGrid/RegularGrid.hpp>
#include <APSS/UpSampler/UpSamplerUnshaken.hpp>
#include <APSS/UpSampler/UpSamplerSimple.hpp>

namespace PointyCloudPlugin
{
    PointyCloudComponent::PointyCloudComponent(const std::string& name, const Ra::Engine::Camera *camera)
        : Ra::Engine::Component( name ), m_camera( camera ),
          m_originalCloud(std::make_shared<PointyCloud>()),
          m_culling(UsefulPointsSelection(m_originalCloud, m_camera)),
          m_selector(std::make_shared<NeighborsSelection>(m_originalCloud, 3)),
          m_projection(OrthogonalProjection(m_selector, m_originalCloud, 3))
    {
        m_upsampler.reset(new UpSamplerUnshaken(3,1));

        // APSS stats
        m_count = 0;
        m_timeCulling = 0;
        m_timeUpsampling = 0;
        m_timeProjecting = 0;
        m_timeLoading = 0;
    }

    PointyCloudComponent::~PointyCloudComponent()
    {
        if(m_count>0)
        {
            LOGP(logINFO) << "\n===Timing computePointyCloud()===\n" <<
                             "Culling    : " << m_timeCulling/m_count << " μs\n" <<
                             "Upsampling : " << m_timeUpsampling/m_count << " μs\n" <<
                             "Projecting : " << m_timeProjecting/m_count << " μs\n" <<
                             "Loading    : " << m_timeLoading/m_count << " μs\n";

        }
        if(m_projection.getCount()>0)
        {
            LOGP(logINFO) << "\n===Timing project()===\n" <<
                             "Neighbors query  : " << m_projection.getTimeNeighbors() << " μs\n" <<
                             "Sphere fitting   : " << m_projection.getTimeFitting() << " μs\n" <<
                             "Point projection : " << m_projection.getTimeProjecting() << " μs" ;
        }
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
            LOGP(logINFO) << "cloud " << cloudName << " has no color. Creation of colors.";
            colors.resize(vertices.size(), Ra::Core::Color::Ones());
        }

        m_workingCloud->addData( Ra::Engine::Mesh::VERTEX_COLOR, colors);

        auto config = Ra::Engine::ShaderConfigurationFactory::getConfiguration("Pointy");

        Ra::Engine::RenderObject* ro = Ra::Engine::RenderObject::createRenderObject(roName, this, Ra::Engine::RenderObjectType::Pointy, m_workingCloud, config);

        m_meshIndex = addRenderObject(ro);

        // init of APSS
        auto sys = static_cast<PointyCloudSystem*>(m_system);

        m_originalCloud->loadFromMesh(m_workingCloud.get());
        setSplatRadius(sys->getSplatRadius());
        setEligible();

        LOGP(logINFO) << "cloud " << cloudName << " has " << m_originalCloud->m_points.size() << " point(s).";
//        LOGP(logINFO) << "cloud " << cloudName << " is pointy at " << m_originalCloud->m_points.size() << " level(s).";

        setUpsamplingMethod(sys->getUpsamplingMethod());
        setProjectionMethod(sys->getProjectionMethod());
        setOptimizationByOctree(sys->isOptimizedByOctree());
        setOptimizationByCUDA(sys->isOptimizedByCUDA());

// FOR APSS TEST (comment computePointyCloud's body)
//        PointyCloud points = m_culling.selectUsefulPoints();
//        PointyCloud points = *m_originalCloud.get();
//        m_upsampler->upSampleCloud(points);
//        m_projection.project(points);
//        points.loadToMesh(m_workingCloud.get());
    }

    void PointyCloudComponent::computePointyCloud()
    {
        PointyCloud points = m_culling.selectUsefulPoints();
        m_upsampler->upSampleCloud(points);
        m_projection.project(points);
        points.loadToMesh(m_workingCloud.get());
    }

    // record timing statistics printed in destructor
//    void PointyCloudComponent::computePointyCloud()
//    {
//        Ra::Core::Timer::TimePoint t0, t1, t2, t3, t4;

//        t0 = Ra::Core::Timer::Clock::now();
//        PointyCloud points = m_culling.selectUsefulPoints();
//        t1 = Ra::Core::Timer::Clock::now();
//        m_upsampler->upSampleCloud(points);
//        t2 = Ra::Core::Timer::Clock::now();
//        m_projection.project(points);
//        t3 = Ra::Core::Timer::Clock::now();
//        points.loadToMesh(m_workingCloud.get());
//        t4 = Ra::Core::Timer::Clock::now();

//        m_timeCulling += Ra::Core::Timer::getIntervalMicro(t0, t1);
//        m_timeUpsampling += Ra::Core::Timer::getIntervalMicro(t1, t2);
//        m_timeProjecting += Ra::Core::Timer::getIntervalMicro(t2, t3);
//        m_timeLoading += Ra::Core::Timer::getIntervalMicro(t3, t4);
//        ++m_count;
//    }

    void PointyCloudComponent::setEligible() {
        for (auto it = m_originalCloud->m_points.begin(); it != m_originalCloud->m_points.end(); ++it) {
            // defined by Patate
            it->setEligible(m_selector->getNeighbors(*it).size() > 6);
        }
    }

    void PointyCloudComponent::setInfluenceRadius(Scalar influenceRadius) {
        m_projection.setInfluenceRadius(influenceRadius);
        m_selector->setInfluenceRadius(influenceRadius);
        m_upsampler->setRadius(influenceRadius);
        setEligible();
    }

    void PointyCloudComponent::setSplatRadius(Scalar splatRadius) {
        for (auto it = m_originalCloud->m_points.begin(); it != m_originalCloud->m_points.end(); ++it) {
            it->setSplatSize(splatRadius);
        }
    }

    void PointyCloudComponent::setBeta(Scalar beta) {
        // TODO donner beta à la projection
    }

    void PointyCloudComponent::setThreshold(int threshold) {
        // TODO donner le threshold a quelqu'un
    }

    void PointyCloudComponent::setM(int M) {
        // TODO donner au ré-echantillonnage
        if (m_upsamplingMethod == FIXED_METHOD) {
            static_cast<UpSamplerUnshaken*>(m_upsampler.get())->setM(M);
        }
        else
        {
            LOGP(logDEBUG)
                << "The Fixed method isn't currently the activated one for the upsampler. "
                << "Impossible to set M.";
        }
    }

    void PointyCloudComponent::setUpsamplingMethod(UPSAMPLING_METHOD method) {
        m_upsamplingMethod = method;
        // TODO switcher entre les méthodes
        auto sys = static_cast<PointyCloudSystem*>(m_system);
        if ( m_upsamplingMethod == FIXED_METHOD ){
            m_upsampler.reset(new UpSamplerUnshaken(sys->getM(), sys->getInfluenceRadius()));
        }
        else if ( m_upsamplingMethod == SIMPLE_METHOD )
            m_upsampler.reset(new UpSamplerSimple(sys->getInfluenceRadius(),sys->getThreshold(),*m_camera));
    }

    void PointyCloudComponent::setProjectionMethod(PROJECTION_METHOD method) {
        m_projectionMethod = method;
        // TODO switcher entre les méthodes
        auto sys = static_cast<PointyCloudSystem*>(m_system);
        m_projection = OrthogonalProjection(m_selector, m_originalCloud, sys->getInfluenceRadius());
    }

    void PointyCloudComponent::setOptimizationByOctree(bool octree) {
        // TODO vérifier que la valeur change bien avant de swaper (risque de recréer la regular grid pour rien)
        auto sys = static_cast<PointyCloudSystem*>(m_system);

        if(octree)
        {
            m_selector.reset(new NeighborsSelectionWithRegularGrid(m_originalCloud, sys->getInfluenceRadius()));
            LOG(logINFO) << "Regular built in " << std::static_pointer_cast<NeighborsSelectionWithRegularGrid>(m_selector)->grid()->getBuildTime() <<
                            " seconds";
        }
        else
        {
            m_selector.reset(new NeighborsSelection(m_originalCloud, sys->getInfluenceRadius()));
        }

        setUpsamplingMethod(sys->getUpsamplingMethod());
        setProjectionMethod(sys->getProjectionMethod());
    }

    void PointyCloudComponent::setOptimizationByCUDA(bool cuda) {
        // TODO donner à tout le monde ???
    }

} // namespace PointyCloudPlugin
