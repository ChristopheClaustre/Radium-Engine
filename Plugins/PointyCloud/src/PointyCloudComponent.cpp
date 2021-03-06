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
          m_selector(std::make_shared<NeighborsSelection>(m_originalCloud, 3)),
          m_projection(OrthogonalProjection(m_selector, m_originalCloud, 3))
    {
        m_upsampler.reset(new UpSamplerUnshaken(m_originalCloud,1));

        // APSS stats
        ON_TIMED(
        m_count = 0;
        m_timeCulling = 0;
        m_timeUpsampling = 0;
        m_timeProjecting = 0;
        m_timeLoading = 0;)
    }

    PointyCloudComponent::~PointyCloudComponent()
    {
        ON_TIMED(
            if(m_count>0)
            {
                LOGP(logINFO)
                    << "\n===Timing computePointyCloud() - " << m_cloudName << "===\n"
                    << "Culling    : " << m_timeCulling/m_count << " μs\n"
                    << "Upsampling : " << m_timeUpsampling/m_count << " μs\n"
                    << "Projecting : " << m_timeProjecting/m_count << " μs\n"
                    << "Loading    : " << m_timeLoading/m_count << " μs\n"
                    << "Total      : " << (m_timeCulling+m_timeUpsampling+m_timeProjecting+m_timeLoading)/m_count
                                                << " μs\n";
            }
            if(m_projection.getCount()>0)
            {
                LOGP(logINFO)
                    << "\n===Timing project() - " << m_cloudName << "===\n"
                    << "Neighbors query  : " << m_projection.getTimeNeighbors() << " μs\n"
                    << "Sphere fitting   : " << m_projection.getTimeFitting() << " μs\n"
                    << "Point projection : " << m_projection.getTimeProjecting() << " μs\n"
                    << "Mean projection number : " << m_projection.getMeanProjectionCount();
            }
        )
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

        m_cloudName = name;
        m_cloudName.append( "_Cloud" );

        m_contentName = data->getName();

        m_workingCloud = Ra::Core::make_shared<Ra::Engine::Mesh>( m_cloudName, GL_POINTS );

        Ra::Core::Transform T = data->getFrame();
        Ra::Core::Transform N;
        N.matrix() = (T.matrix()).inverse().transpose();

        Ra::Core::Vector3Array vertices;
        Ra::Core::Vector3Array normals;

        for ( const auto& v : data->getVertices() ) vertices.push_back( T * v );
        for ( const auto& n : data->getNormals() )  normals.push_back( (N * n).normalized() );

        m_workingCloud->loadPointyGeometry( vertices, normals );

        Ra::Core::Vector4Array colors;
        if (data->hasColors()) {
            for ( const auto& v : data->getColors() ) colors.push_back( v );
        }
        else {
            LOGP(logINFO) << "cloud " << m_cloudName << " has no color. Creation of colors.";
            colors.resize(vertices.size(), Ra::Core::Color::Ones());
        }
        Ra::Core::Vector1Array radiuses;
        radiuses.resize(vertices.size(), 1.0);

        m_workingCloud->addData( Ra::Engine::Mesh::VERTEX_COLOR, colors);
        m_workingCloud->addData( Ra::Engine::Mesh::POINT_RADIUS, radiuses);

        auto config = Ra::Engine::ShaderConfigurationFactory::getConfiguration("Pointy");

        Ra::Engine::RenderObject* ro = Ra::Engine::RenderObject::createRenderObject(roName, this, Ra::Engine::RenderObjectType::Pointy, m_workingCloud, config);

        m_meshIndex = addRenderObject(ro);

        // init of APSS
        auto sys = static_cast<PointyCloudSystem*>(m_system);

        m_originalCloud->loadFromMesh(m_workingCloud.get());

        setSplatRadius(sys->getSplatRadius());
        m_culling = new UsefulPointsSelection(m_originalCloud, m_camera);

        LOGP(logINFO) << "cloud " << m_cloudName << " has " << m_originalCloud->size() << " point(s).";

        setUpsamplingMethod(sys->getUpsamplingMethod());
        setProjectionMethod(sys->getProjectionMethod());
        setOptimizationByOctree(sys->isOptimizedByOctree());
        setOptimizationByCUDA(sys->isOptimizedByCUDA());
        setEligibleFlags();


// FOR APSS TEST (comment computePointyCloud's body)
//        PointyCloud points = m_culling.selectUsefulPoints();
//        PointyCloud points = *m_originalCloud.get();
//        m_upsampler->upSampleCloud(points);
//        m_projection.project(points);
//        points.loadToMesh(m_workingCloud.get());
    }

    void PointyCloudComponent::computePointyCloud()
    {

        ON_TIMED(auto t0 = Ra::Core::Timer::Clock::now());
        m_culling->selectUsefulPoints();
        ON_TIMED(auto t1 = Ra::Core::Timer::Clock::now());
        m_upsampler->upSampleCloud(m_culling->getIndices(), m_culling->getN());
        ON_TIMED(auto t2 = Ra::Core::Timer::Clock::now());
        PointyCloud& aCloud = m_upsampler->getUpsampledCloud();
        m_projection.project(aCloud);
        ON_TIMED(auto t3 = Ra::Core::Timer::Clock::now());
        aCloud.loadToMesh(m_workingCloud.get());
        ON_TIMED(auto t4 = Ra::Core::Timer::Clock::now());

        ON_TIMED(
        m_timeCulling += Ra::Core::Timer::getIntervalMicro(t0, t1);
        m_timeUpsampling += Ra::Core::Timer::getIntervalMicro(t1, t2);
        m_timeProjecting += Ra::Core::Timer::getIntervalMicro(t2, t3);
        m_timeLoading += Ra::Core::Timer::getIntervalMicro(t3, t4);
        ++m_count;)
    }

    void PointyCloudComponent::setEligibleFlags() {
        #pragma omp parallel for
        for (int i = 0; i < m_originalCloud->size(); ++i) {
            m_originalCloud->at(i).eligible() = (m_selector->isEligible(m_originalCloud->at(i)));
        }
    }

    void PointyCloudComponent::setInfluenceRadius(Scalar influenceRadius) {
        m_projection.setInfluenceRadius(influenceRadius);
        m_selector->setInfluenceRadius(influenceRadius);
        setEligibleFlags();
        m_upsampler->resetUpsamplingInfo();
    }

    void PointyCloudComponent::setSplatRadius(Scalar splatRadius) {
        #pragma omp parallel for
        for (int i = 0; i < m_originalCloud->size(); ++i) {
            m_originalCloud->at(i).radius() = splatRadius;
        }

        auto sys = static_cast<PointyCloudSystem*>(m_system);
        if (!sys->isAPSSused()) {
            resetWorkingCloud();
        }

        m_upsampler->resetUpsamplingInfo();
    }

    void PointyCloudComponent::setThreshold(int threshold) {
        if (m_upsamplingMethod != FIXED_METHOD) {
            static_cast<UpSamplerSimple*>(m_upsampler.get())->setThreshold(threshold);
        }
        else {
            LOGP(logDEBUG)
                << "The Fixed method IS currently the activated one for the upsampler. Impossible to set threshold.";
        }
    }

    void PointyCloudComponent::setM(int M) {
        if (m_upsamplingMethod == FIXED_METHOD) {
            static_cast<UpSamplerUnshaken*>(m_upsampler.get())->setM(M);
        }
        else
        {
            LOGP(logDEBUG)
                << "The Fixed method isn't currently the activated one for the upsampler. Impossible to set M.";
        }
    }

    void PointyCloudComponent::setUpsamplingMethod(UPSAMPLING_METHOD method) {
        m_upsamplingMethod = method;

        auto sys = static_cast<PointyCloudSystem*>(m_system);

        if ( m_upsamplingMethod == FIXED_METHOD ){
            m_upsampler.reset(new UpSamplerUnshaken(m_originalCloud, sys->getM()));
        }
        else if ( m_upsamplingMethod == SIMPLE_METHOD ){
            m_upsampler.reset(new UpSamplerSimple(m_originalCloud, sys->getThreshold(), *m_camera));
        }
        // TODO the last method (but not the least)
    }

    void PointyCloudComponent::setProjectionMethod(PROJECTION_METHOD method) {
        m_projectionMethod = method;
        // TODO choosing between all the methods
        auto sys = static_cast<PointyCloudSystem*>(m_system);
        m_projection = OrthogonalProjection(m_selector, m_originalCloud, sys->getInfluenceRadius());
    }

    void PointyCloudComponent::setOptimizationByOctree(bool octree) {
        // TODO vérifier que la valeur change bien avant de swaper (risque de recréer la regular grid pour rien)
        auto sys = static_cast<PointyCloudSystem*>(m_system);

        if(octree)
        {
            m_selector.reset(new NeighborsSelectionWithRegularGrid(m_originalCloud, sys->getInfluenceRadius()));
            LOGP(logINFO) << m_cloudName << "'s Regular grid built in "
                         << std::static_pointer_cast<NeighborsSelectionWithRegularGrid>(m_selector)->grid()->getBuildTime()
                         << " seconds";
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

    void PointyCloudComponent::resetWorkingCloud()
    {
        m_originalCloud->loadToMesh(m_workingCloud.get());
    }

} // namespace PointyCloudPlugin
