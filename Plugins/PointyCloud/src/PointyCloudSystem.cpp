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

#include <PointyCloudComponent.hpp>

namespace PointyCloudPlugin
{

    PointyCloudSystem::PointyCloudSystem()
        : Ra::Engine::System()
    {
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
            std::string componentName = "PCC_" + entity->getName() + std::to_string( id++ );
            PointyCloudComponent * comp = new PointyCloudComponent( componentName );
            entity->addComponent( comp );
            comp->handleMeshLoading(data);
            registerComponent( entity, comp );
        }
    }

    void PointyCloudSystem::generateTasks( Ra::Core::TaskQueue* taskQueue, const Ra::Engine::FrameInfo& frameInfo )
    {
        // TODO générer la computePointyCloudTask ;) ;) ;) #YOLO #SWAG
    }

    void PointyCloudSystem::setSplatRadius(float splatRadius)
    {
        m_splatRadius = splatRadius;
        // TODO donner au renderer
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

    void PointyCloudSystem::setThreshold(float threshold)
    {
        m_threshold = threshold;
        // TODO donner à tous les components
    }

    void PointyCloudSystem::setUpsamplingMethod(UPSAMPLING_METHOD upsampler)
    {
        m_upsampler = upsampler;
        // TODO donner à tous les components
    }

    void PointyCloudSystem::setProjectionMethod(PROJECTION_METHOD projector)
    {
        m_projector = projector;
        // TODO donner à tous les components
    }

    void PointyCloudSystem::setOptimizationByOctree(bool octree)
    {
        m_octree = octree;
        // TODO donner à tous les components
    }

    void PointyCloudSystem::setOptimizationByCUDA(bool cuda)
    {
        m_cuda = cuda;
        // TODO donner à tous les components
    }

} // namespace PointyCloudPlugin
