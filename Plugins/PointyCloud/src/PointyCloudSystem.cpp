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

#include <Cuda/APSSTask.hpp>

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
          m_APSS(true), m_rendererUsed(true)
    {
        m_renderer = new PointyCloudPlugin::PointyCloudRenderer(m_viewer->width(), m_viewer->height(), m_splatRadius);
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

                // Cuda APSS
                for(auto& roIdx : comp->m_renderObjects)
                {
                    std::shared_ptr<Ra::Engine::Mesh> mesh = Ra::Engine::RadiumEngine::getInstance()->getRenderObjectManager()->getRenderObject(roIdx)->getMesh();
                    Cuda::APSS* apss = new Cuda::APSS(mesh->getGeometry().m_vertices.data(),
                                                      mesh->getGeometry().m_normals.data(),
                                                      mesh->getData(Ra::Engine::Mesh::VERTEX_COLOR).data(),
                                                      mesh->getGeometry().m_vertices.size());
                    m_APPS.push_back(apss);
                    m_mesh.push_back(mesh);
                }
            }
        }

    }

    std::vector<PointyCloudComponent*> PointyCloudSystem::getComponents()
    {
        return pointyCloudComponentList;
    }

    void PointyCloudSystem::generateTasks( Ra::Core::TaskQueue* taskQueue, const Ra::Engine::FrameInfo& frameInfo )
    {
        for(int k=0; k<m_APPS.size(); ++k)
            taskQueue->registerTask(new APSSTask(m_APPS[k], m_mesh[k], m_viewer->getCameraInterface()->getCamera(), m_splatRadius));

//        if(m_APSS)
//            for(auto comp : pointyCloudComponentList)
//                taskQueue->registerTask(new ComputePointyCloudTask(comp));
    }

    void PointyCloudSystem::setSplatRadius(Scalar splatRadius)
    {
        m_splatRadius = splatRadius;
        //TODO(xavier)Envoi au renderer à retirer
        m_renderer->setSplatSize(splatRadius);
        for (auto comp : pointyCloudComponentList) {
            comp->setSplatRadius(splatRadius);
        }
    }

    void PointyCloudSystem::setInfluenceRadius(Scalar influenceRadius)
    {
        m_influenceRadius = influenceRadius;
        for (auto comp : pointyCloudComponentList) {
            comp->setInfluenceRadius(influenceRadius);
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
        for (auto comp : pointyCloudComponentList) {
            comp->setM(M);
        }
    }

    void PointyCloudSystem::setUpsamplingMethod(UPSAMPLING_METHOD upsampler)
    {
        m_upsampler = upsampler;
        for (auto comp : pointyCloudComponentList) {
            comp->setUpsamplingMethod(upsampler);
        }
    }

    void PointyCloudSystem::setProjectionMethod(PROJECTION_METHOD projector)
    {
        m_projector = projector;
        for (auto comp : pointyCloudComponentList) {
            comp->setProjectionMethod(projector);
        }
    }

    void PointyCloudSystem::setOptimizationByOctree(bool octree)
    {
        m_octree = octree;
        for (auto comp : pointyCloudComponentList) {
            comp->setOptimizationByOctree(octree);
        }
    }

    void PointyCloudSystem::setOptimizationByCUDA(bool cuda)
    {
        m_cuda = cuda;
        for (auto comp : pointyCloudComponentList) {
            comp->setOptimizationByCUDA(cuda);
        }
    }

    void PointyCloudSystem::setAPSS(bool apss)
    {
        m_APSS = apss;
        if(!apss)
            for (auto comp : pointyCloudComponentList)
                comp->resetOriginalCloud();
    }

    void PointyCloudSystem::setRenderer(bool renderer)
    {
        m_rendererUsed = renderer;
        if(renderer)
            m_viewer->changeRenderer(m_rendererIndex);
        else
            m_viewer->changeRenderer(0);
    }

} // namespace PointyCloudPlugin
