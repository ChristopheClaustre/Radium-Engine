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
#include <GuiBase/Viewer/Viewer.hpp>

#include <PointyCloudComponent.hpp>
#include <Renderer/PointyCloudRenderer.hpp>
#include <ComputePointyCloudTask.hpp>

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
          m_octree(false), m_cuda(false),
          m_APSS(true), m_rendererUsed(true), to_refresh(true)
    {
        m_renderer = new PointyCloudPlugin::PointyCloudRenderer(m_viewer->width(), m_viewer->height());
        m_rendererIndex = m_viewer->addRenderer(m_renderer);
        setRenderer(true);

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
            bool valid = data->hasVertices() && data->hasNormals();

            if (data->getType() != Ra::Asset::GeometryData::POINT_CLOUD) {
                if (valid) {
                    LOGP(logINFO) << data->getName() << " isn't a Point cloud, but it can be loaded as one.";
                }
                else {
                    LOGP(logINFO) << data->getName() << " isn't a Point cloud, and it CAN'T be loaded as one (no normal nor position).";
                }
            }
            else {
                if (!valid) {
                    LOGP(logINFO) << "Failed to load " << data->getName() << " : cloud has no normal nor position and it's needed for APSS.";
                }
            }

            if (valid) {
                std::string componentName = "PointyC_" + entity->getName() + std::to_string( id++ );
                PointyCloudComponent * comp = new PointyCloudComponent( componentName, m_viewer->getCameraInterface()->getCamera() );
                entity->addComponent( comp );
                registerComponent( entity, comp );
                comp->handlePointyCloudLoading(data);

                pointyCloudComponentList.push_back(comp);
            }
        }

    }

    std::vector<PointyCloudComponent*> PointyCloudSystem::getComponents()
    {
        return pointyCloudComponentList;
    }

    void PointyCloudSystem::generateTasks( Ra::Core::TaskQueue* taskQueue, const Ra::Engine::FrameInfo& frameInfo )
    {
        if(m_APSS) {
            Ra::Core::Matrix4 currProj = m_viewer->getCameraInterface()->getProjMatrix();
            Ra::Core::Matrix4 currView = m_viewer->getCameraInterface()->getViewMatrix();
            if (currProj != m_prevProjMatrix || currView != m_prevViewMatrix || to_refresh) {

                for(auto * comp : pointyCloudComponentList) {
                    taskQueue->registerTask(new ComputePointyCloudTask(comp));
                }

                m_prevProjMatrix = currProj;
                m_prevViewMatrix = currView;

                to_refresh = false;
            }
        }
    }

    void PointyCloudSystem::setSplatRadius(Scalar splatRadius)
    {
        m_splatRadius = splatRadius;

        for (auto comp : pointyCloudComponentList) {
            comp->setSplatRadius(splatRadius);
        }
        to_refresh = true;
    }

    void PointyCloudSystem::setInfluenceRadius(Scalar influenceRadius)
    {
        m_influenceRadius = influenceRadius;
        for (auto comp : pointyCloudComponentList) {
            comp->setInfluenceRadius(influenceRadius);
        }
        to_refresh = true;
    }

    void PointyCloudSystem::setThreshold(int threshold)
    {
        m_threshold = threshold;
        // TODO donner à tout les components
        for (auto comp : pointyCloudComponentList) {
            comp->setThreshold(threshold);
        }
        to_refresh = true;
    }

    void PointyCloudSystem::setM(int M)
    {
        m_M = M;
        for (auto comp : pointyCloudComponentList) {
            comp->setM(M);
        }
        to_refresh = true;
    }

    void PointyCloudSystem::setUpsamplingMethod(UPSAMPLING_METHOD upsampler)
    {
        m_upsampler = upsampler;
        for (auto comp : pointyCloudComponentList) {
            comp->setUpsamplingMethod(upsampler);
        }
        to_refresh = true;
    }

    void PointyCloudSystem::setProjectionMethod(PROJECTION_METHOD projector)
    {
        m_projector = projector;
        for (auto comp : pointyCloudComponentList) {
            comp->setProjectionMethod(projector);
        }
        to_refresh = true;
    }

    void PointyCloudSystem::setOptimizationByOctree(bool octree)
    {
        m_octree = octree;
        for (auto comp : pointyCloudComponentList) {
            comp->setOptimizationByOctree(octree);
        }
        to_refresh = true;
    }

    void PointyCloudSystem::setOptimizationByCUDA(bool cuda)
    {
        m_cuda = cuda;
        for (auto comp : pointyCloudComponentList) {
            comp->setOptimizationByCUDA(cuda);
        }
        to_refresh = true;
    }

    void PointyCloudSystem::setAPSS(bool apss)
    {
        m_APSS = apss;
        if(!apss)
        {
            for (auto comp : pointyCloudComponentList)
            {
                comp->resetOriginalCloud();
            }
        }
        to_refresh = true;
    }

    void PointyCloudSystem::setRenderer(bool renderer)
    {
        m_rendererUsed = renderer;
        if(renderer) {
            m_viewer->changeRenderer(m_rendererIndex);
        }
        else
        {
            m_viewer->changeRenderer(0);
        }
        to_refresh = true;
    }

} // namespace PointyCloudPlugin
