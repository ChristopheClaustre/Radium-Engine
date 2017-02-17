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
          m_splatRadius(PointyCloudPluginC::splatRadiusInit.init),
          m_influenceRadius(PointyCloudPluginC::influenceInit.init),
          m_beta(PointyCloudPluginC::betaInit.init),
          m_threshold(PointyCloudPluginC::thresholdInit.init),
          m_M(PointyCloudPluginC::mInit.init),
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
            if (data->hasNormals()) {
                std::string componentName = "PointyC_" + entity->getName() + std::to_string( id++ );
                PointyCloudComponent * comp = new PointyCloudComponent( componentName, m_viewer->getCameraInterface()->getCamera() );
                entity->addComponent( comp );
                registerComponent( entity, comp );
                comp->handlePointyCloudLoading(data);

                pointyCloudComponentList.push_back(comp);
            }
            else {
                LOGP(logINFO) << "Failed to load " << data->getName() << " : cloud has no normal.";
            }
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

    void PointyCloudSystem::setSplatRadius(Scalar splatRadius)
    {
        m_splatRadius = splatRadius;
        m_renderer->setSplatSize(splatRadius);
    }

    void PointyCloudSystem::setInfluenceRadius(Scalar influenceRadius)
    {
        m_influenceRadius = influenceRadius;
        // TODO donner à tous les components
        for (auto comp : pointyCloudComponentList) {
            comp->setInfluenceRadius(influenceRadius);
        }
    }
    void PointyCloudSystem::setBeta(Scalar beta)
    {
        m_beta = beta;
        // TODO donner à tous les components
        for (auto comp : pointyCloudComponentList) {
            comp->setBeta(beta);
        }
    }

    void PointyCloudSystem::setThreshold(int threshold)
    {
        m_threshold = threshold;
        // TODO donner à tout les components
        for (auto comp : pointyCloudComponentList) {
            comp->setThreshold(threshold);
        }
    }

    void PointyCloudSystem::setM(int M)
    {
        m_M = M;
        // TODO donner à tout les components
        for (auto comp : pointyCloudComponentList) {
            comp->setM(M);
        }
    }

    void PointyCloudSystem::setUpsamplingMethod(UPSAMPLING_METHOD upsampler)
    {
        m_upsampler = upsampler;
        // TODO donner à tout les components
        for (auto comp : pointyCloudComponentList) {
            comp->setUpsamplingMethod(upsampler);
        }
    }

    void PointyCloudSystem::setProjectionMethod(PROJECTION_METHOD projector)
    {
        m_projector = projector;
        // TODO donner à tout les components
        for (auto comp : pointyCloudComponentList) {
            comp->setProjectionMethod(projector);
        }
    }

    void PointyCloudSystem::setOptimizationByOctree(bool octree)
    {
        m_octree = octree;
        // TODO donner à tout les components
        for (auto comp : pointyCloudComponentList) {
            comp->setOptimizationByOctree(octree);
        }
    }

    void PointyCloudSystem::setOptimizationByCUDA(bool cuda)
    {
        m_cuda = cuda;
        // TODO donner à tout les components
        for (auto comp : pointyCloudComponentList) {
            comp->setOptimizationByCUDA(cuda);
        }
    }

} // namespace PointyCloudPlugin
