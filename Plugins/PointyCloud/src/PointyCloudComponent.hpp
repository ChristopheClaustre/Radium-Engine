#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDCOMPONENT_HPP
#define POINTYCLOUDPLUGIN_POINTYCLOUDCOMPONENT_HPP

#include <PointyCloudPlugin.hpp>

#include <Core/Mesh/MeshTypes.hpp>
#include <Core/Mesh/TriangleMesh.hpp>

#include <Engine/Component/Component.hpp>

namespace Ra
{
    namespace Engine
    {
        struct RenderTechnique;
        class Mesh;
    }

    namespace Asset
    {
        class GeometryData;
    }
}

namespace PointyCloudPlugin
{
    class POINTY_PLUGIN_API PointyCloudComponent : public Ra::Engine::Component
    {
    public:
        PointyCloudComponent( const std::string& name);
        virtual ~PointyCloudComponent();


        virtual void initialize() override;

        void handleMeshLoading(const Ra::Asset::GeometryData* data);

        /// Returns the index of the associated RO (the display mesh)
        Ra::Core::Index getRenderObjectIndex() const;

        /// Returns the current display geometry.
        const Ra::Core::TriangleMesh& getMesh() const;

        void setInfluenceRadius(float);
        void setBeta(float);
        void setThreshold(float);
        void setUpsamplingMethod(UPSAMPLING_METHOD);
        void setProjectionMethod(PROJECTION_METHOD);
        void setOptimizationByOctree(bool);
        void setOptimizationByCUDA(bool);

    private:

        const Ra::Engine::Mesh& getDisplayMesh() const;
        Ra::Engine::Mesh& getDisplayMesh();

        // Pointy cloud accepts to give its mesh and (if deformable) to update it
        const Ra::Core::TriangleMesh* getMeshOutput() const;
        void setMeshInput(const std::shared_ptr<Ra::Engine::Mesh> meshShared );

        const Ra::Core::Index* roIndexRead() const;

    private:
        Ra::Core::Index m_meshIndex;
        Ra::Core::Index m_aabbIndex;
        std::string m_contentName;
    };

} // namespace PointyCloudPlugin

#endif // POINTYCLOUDPLUGIN_POINTYCLOUDCOMPONENT_HPP
