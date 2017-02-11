#include <PointyCloudSystem.hpp>

#include <Core/String/StringUtils.hpp>
#include <Core/Tasks/Task.hpp>
#include <Core/Tasks/TaskQueue.hpp>

#include <Engine/RadiumEngine.hpp>
#include <Engine/Entity/Entity.hpp>
#include <Engine/FrameInfo.hpp>
#include <Engine/Renderer/RenderTechnique/RenderTechnique.hpp>
#include <Engine/Assets/FileData.hpp>
#include <Engine/Assets/GeometryData.hpp>
#include <Engine/Managers/ComponentMessenger/ComponentMessenger.hpp>

#include <GuiBase/Viewer/CameraInterface.hpp>

#include <PointyCloudComponent.hpp>
#include <PointyCloudPlugin.hpp>

namespace PointyCloudPlugin
{

    PointyCloudSystem::PointyCloudSystem(Ra::Gui::Viewer *viewer)
        : Ra::Engine::System(), m_viewer(viewer),
          m_splatRadius(PointyCloudPluginC::splatRadiusInit.min),
          m_influenceRadius(PointyCloudPluginC::influenceInit.min),
          m_beta(PointyCloudPluginC::betaInit.min),
          m_threshold(PointyCloudPluginC::thresholdInit.min),
          m_M(PointyCloudPluginC::mInit.min),
          m_upsampler(FIXED_METHOD), m_projector(ORTHOGONAL_METHOD),
          m_octree(false), m_cuda(false)
    {
        m_renderer = new PointyCloudPlugin::PointyCloudRenderer(m_viewer->width(), m_viewer->height(), m_splatRadius);
        m_rendererIndex = m_viewer->addRenderer(m_renderer);
        m_viewer->changeRenderer(m_rendererIndex);

        Ra::Engine::ShaderConfiguration config("Pointy");
        config.addShader(Ra::Engine::ShaderType_VERTEX,   "../Shaders/Pointy/PointyQuad.vert.glsl");
        config.addShader(Ra::Engine::ShaderType_GEOMETRY, "../Shaders/Pointy/PointyQuad.geom.glsl");
        config.addShader(Ra::Engine::ShaderType_FRAGMENT, "../Shaders/Pointy/PointyQuad.frag.glsl");
        Ra::Engine::ShaderConfigurationFactory::addConfiguration("Pointy", config);
    }

    PointyCloudSystem::~PointyCloudSystem()
    {
    }

    void PointyCloudSystem::handleAssetLoading( Ra::Engine::Entity* entity, const Ra::Asset::FileData* fileData )
    {
        auto geomData = fileData->getGeometryData();

        uint id = 0;

        for ( const auto& data : geomData )
        {
            std::string componentName = "PointyC_" + entity->getName() + std::to_string( id++ );
            PointyCloudComponent * comp = new PointyCloudComponent( componentName, m_viewer->getCameraInterface()->getCamera() );
            entity->addComponent( comp );
            comp->handlePointyCloudLoading(data);
            registerComponent( entity, comp );
            //(xavier) Ajout du nouveau component dans la liste des components
            pointyCloudComponentList.push_back(comp);
        }

    }

    std::vector<PointyCloudComponent*> PointyCloudSystem::getComponents()
    {
        return pointyCloudComponentList;
    }

    void PointyCloudSystem::generateTasks( Ra::Core::TaskQueue* taskQueue, const Ra::Engine::FrameInfo& frameInfo )
    {
        ComputePointyCloudTask* task = new ComputePointyCloudTask(this);
        taskQueue->registerTask(task);
    }

    void PointyCloudSystem::setSplatRadius(float splatRadius)
    {
        m_splatRadius = splatRadius;
        m_renderer->setSplatSize(splatRadius);
    }

    void PointyCloudSystem::setInfluenceRadius(float influenceRadius)
    {
        m_influenceRadius = influenceRadius;
        // TODO donner à tous les components
    }
    void PointyCloudSystem::setBeta(float beta)
    {
        m_beta = beta;
        // TODO donner à tous les components
    }

    void PointyCloudSystem::setThreshold(int threshold)
    {
        m_threshold = threshold;
        // TODO donner à tout les components
    }

    void PointyCloudSystem::setM(int M)
    {
        m_M = M;
        // TODO donner à tout les components
    }

    void PointyCloudSystem::setUpsamplingMethod(UPSAMPLING_METHOD upsampler)
    {
        m_upsampler = upsampler;
        // TODO donner à tout les components
    }

    void PointyCloudSystem::setProjectionMethod(PROJECTION_METHOD projector)
    {
        m_projector = projector;
        // TODO donner à tout les components
    }

    void PointyCloudSystem::setOptimizationByOctree(bool octree)
    {
        m_octree = octree;
        // TODO donner à tout les components
    }

    void PointyCloudSystem::setOptimizationByCUDA(bool cuda)
    {
        m_cuda = cuda;
        // TODO donner à tout les components
    }

} // namespace PointyCloudPlugin
