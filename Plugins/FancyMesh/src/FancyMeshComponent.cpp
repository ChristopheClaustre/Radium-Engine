#include <FancyMeshComponent.hpp>

#include <iostream>

#include <Core/Mesh/MeshUtils.hpp>
#include <Core/String/StringUtils.hpp>

#include <Core/Geometry/Normal/Normal.hpp>

#include <Engine/Managers/ComponentMessenger/ComponentMessenger.hpp>
#include <Engine/Renderer/RenderObject/RenderObjectManager.hpp>

#include <Engine/Managers/AssetManager.hpp>
#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <Engine/Renderer/RenderObject/Primitives/DrawPrimitives.hpp>
#include <Engine/Renderer/RenderObject/RenderObject.hpp>
#include <Engine/Renderer/RenderObject/RenderObjectTypes.hpp>
#include <Engine/Renderer/RenderTechnique/RenderTechnique.hpp>
#include <Engine/Renderer/RenderTechnique/ShaderConfigFactory.hpp>
#include <Engine/Renderer/RenderTechnique/ShaderProgram.hpp>
#include <Engine/Renderer/RenderTechnique/ShaderProgramManager.hpp>

#include <Engine/Assets/FileData.hpp>
#include <Engine/Assets/GeometryData.hpp>

using Ra::Core::TriangleMesh;
using Ra::Engine::ComponentMessenger;

typedef Ra::Core::VectorArray<Ra::Core::Triangle> TriangleArray;

namespace FancyMeshPlugin
{
    FancyMeshComponent::FancyMeshComponent(const std::string& name, bool deformable)
        : Ra::Engine::Component(name)
        , m_deformable(deformable)
    {
    }

    FancyMeshComponent::~FancyMeshComponent() {}

    void FancyMeshComponent::initialize() {}

    void FancyMeshComponent::addMeshRenderObject(const Ra::Core::TriangleMesh& mesh,
                                                 const std::string&            name)
    {
        setupIO(name);

        auto mgr = Ra::Engine::AssetManager::getInstance();

        Ra::Engine::Mesh* displayMesh = mgr->mesh(mgr->createMesh(name));
        displayMesh->loadGeometry(mesh);

        auto renderObject =
            Ra::Engine::RenderObject::createRenderObject(name,
                                                         this,
                                                         Ra::Engine::RenderObjectType::Fancy,
                                                         displayMesh);
        addRenderObject(renderObject);
    }

    void FancyMeshComponent::handleMeshLoading(const Ra::Asset::GeometryData* data)
    {
        using Ra::Engine::RenderObject;

        std::string name(m_name);
        name.append("_" + data->getName());

        std::string roName = name;
        m_contentName      = data->getName();
        RenderObject* ro   = RenderObject::createFancyFromAsset(roName, this, data, true);

        setupIO(data->getName());
        m_meshIndex = addRenderObject(ro);
    }

    Ra::Core::Index FancyMeshComponent::getRenderObjectIndex() const { return m_meshIndex; }

    const Ra::Core::TriangleMesh& FancyMeshComponent::getMesh() const
    {
        return getDisplayMesh().getGeometry();
    }

    void FancyMeshComponent::setupIO(const std::string& id)
    {
        ComponentMessenger::CallbackTypes<TriangleMesh>::Getter cbOut =
            std::bind(&FancyMeshComponent::getMeshOutput, this);
        ComponentMessenger::getInstance()->registerOutput<TriangleMesh>(getEntity(),
                                                                        this,
                                                                        id,
                                                                        cbOut);

        if (m_deformable)
        {
            ComponentMessenger::CallbackTypes<TriangleMesh>::Setter cbIn =
                std::bind(&FancyMeshComponent::setMeshInput, this, std::placeholders::_1);
            ComponentMessenger::getInstance()->registerInput<TriangleMesh>(getEntity(),
                                                                           this,
                                                                           id,
                                                                           cbIn);

            ComponentMessenger::CallbackTypes<Ra::Core::Vector3Array>::ReadWrite vRW =
                std::bind(&FancyMeshComponent::getVerticesRw, this);
            ComponentMessenger::getInstance()
                ->registerReadWrite<Ra::Core::Vector3Array>(getEntity(), this, id + "v", vRW);

            ComponentMessenger::CallbackTypes<Ra::Core::Vector3Array>::ReadWrite nRW =
                std::bind(&FancyMeshComponent::getNormalsRw, this);
            ComponentMessenger::getInstance()
                ->registerReadWrite<Ra::Core::Vector3Array>(getEntity(), this, id + "n", nRW);

            ComponentMessenger::CallbackTypes<TriangleArray>::ReadWrite tRW =
                std::bind(&FancyMeshComponent::getTrianglesRw, this);
            ComponentMessenger::getInstance()->registerReadWrite<TriangleArray>(getEntity(),
                                                                                this,
                                                                                id + "t",
                                                                                tRW);
        }
    }
    const Ra::Engine::Mesh& FancyMeshComponent::getDisplayMesh() const
    {
        return *(getRoMgr()->getRenderObject(getRenderObjectIndex())->mesh);
    }

    Ra::Engine::Mesh& FancyMeshComponent::getDisplayMesh()
    {
        return *(getRoMgr()->getRenderObject(getRenderObjectIndex())->mesh);
    }

    const Ra::Core::TriangleMesh* FancyMeshComponent::getMeshOutput() const { return &(getMesh()); }

    void FancyMeshComponent::setMeshInput(const TriangleMesh* meshptr)
    {
        CORE_ASSERT(meshptr, " Input is null");
        CORE_ASSERT(m_deformable, "Mesh is not deformable");

        Ra::Engine::Mesh& displayMesh = getDisplayMesh();
        displayMesh.loadGeometry(*meshptr);
    }

    Ra::Core::Vector3Array* FancyMeshComponent::getVerticesRw()
    {
        getDisplayMesh().setDirty(Ra::Engine::Mesh::VERTEX_POSITION);
        return &(getDisplayMesh().getGeometry().m_vertices);
    }

    Ra::Core::Vector3Array* FancyMeshComponent::getNormalsRw()
    {
        getDisplayMesh().setDirty(Ra::Engine::Mesh::VERTEX_NORMAL);
        return &(getDisplayMesh().getGeometry().m_normals);
    }

    Ra::Core::VectorArray<Ra::Core::Triangle>* FancyMeshComponent::getTrianglesRw()
    {
        getDisplayMesh().setDirty(Ra::Engine::Mesh::INDEX);
        return &(getDisplayMesh().getGeometry().m_triangles);
    }

    void FancyMeshComponent::rayCastQuery(const Ra::Core::Ray& r) const
    {
        auto result = Ra::Core::MeshUtils::castRay(getMesh(), r);
        int  tidx   = result.m_hitTriangle;
        if (tidx >= 0)
        {
            LOG(logINFO) << " Hit triangle " << tidx;
            LOG(logINFO) << " Nearest vertex " << result.m_nearestVertex;
            LOG(logINFO) << "Hit position : " << r.pointAt(result.m_t).transpose();
        }
    }

} // namespace FancyMeshPlugin
