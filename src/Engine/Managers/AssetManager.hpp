#ifndef RADIUMENGINE_ASSETMANAGER_HPP
#define RADIUMENGINE_ASSETMANAGER_HPP

#include <Engine/RaEngine.hpp>

#include <memory>
#include <string>

#include <Core/Index/IndexMap.hpp>
#include <Core/Utils/Singleton.hpp>

namespace Ra
{
    namespace Engine
    {
        class RenderTechnique;
        class Material;
        class Mesh;
        class ShaderProgram;
        class Texture;

        using MaterialPtr = std::shared_ptr<Material>;
        using RenderTechniquePtr = std::shared_ptr<RenderTechnique>;

        class RA_ENGINE_API AssetManager
        {
            RA_SINGLETON_INTERFACE(AssetManager);

        public:
            AssetManager() = default;
            ~AssetManager() = default;

            Core::Index createRenderTechnique();
            Core::Index createMaterial(std::string name);

            inline RenderTechnique* renderTechnique(const Core::Index& idx)
            { return m_renderTechniques.at(idx).get(); }
            inline const RenderTechnique* renderTechnique(const Core::Index& idx) const
            { return m_renderTechniques.at(idx).get(); }

            inline Material* material(const Core::Index& idx)
            { return m_materials.at(idx).get(); }
            inline const Material* material(const Core::Index& idx) const
            { return m_materials.at(idx).get(); }

            inline void removeRenderTechnique(const Core::Index& idx)
            { m_renderTechniques.remove(idx); }
            inline void removeMaterial(const Core::Index& idx)
            { m_materials.remove(idx); }



        private:
            Core::IndexMap<RenderTechniquePtr> m_renderTechniques;
            Core::IndexMap<MaterialPtr> m_materials;
        };
    }
}

#endif // RADIUMENGINE_ASSETMANAGER_HPP
