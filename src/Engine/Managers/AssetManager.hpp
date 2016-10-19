#ifndef RADIUMENGINE_ASSETMANAGER_HPP
#define RADIUMENGINE_ASSETMANAGER_HPP

#include <Engine/RaEngine.hpp>

#include <memory>
#include <string>

#include <Core/Index/IndexMap.hpp>
#include <Core/Utils/Singleton.hpp>

#include <Engine/Renderer/OpenGL/OpenGL.hpp>

namespace Ra
{
    namespace Engine
    {
        class RenderTechnique;
        class Material;
        class Mesh;
        class ShaderProgram;
        class ShaderConfiguration;
        class Texture;

        using MaterialPtr           = std::shared_ptr<Material>;
        using RenderTechniquePtr    = std::shared_ptr<RenderTechnique>;
        using MeshPtr               = std::shared_ptr<Mesh>;
        using TexturePtr            = std::shared_ptr<Texture>;
        using ShaderProgramPtr      = std::shared_ptr<ShaderProgram>;

        class RA_ENGINE_API AssetManager
        {
            RA_SINGLETON_INTERFACE(AssetManager);

        public:
            AssetManager() = default;
            ~AssetManager() = default;

            Core::Index createRenderTechnique();
            Core::Index createMaterial(std::string name);

            Core::Index createShaderProgram();
            Core::Index createShaderProgram(std::string vertFile,
                                            std::string fragFile,
                                            std::string geomFile = "",
                                            std::string tescFile = "",
                                            std::string teseFile = "");
            Core::Index createShaderProgram(std::string compFile);
            Core::Index createShaderProgram(const ShaderConfiguration& config);

            void reloadShaderPrograms();

            Core::Index createTexture(std::string name);
            Core::Index createMesh(std::string name, GLenum renderMode = GL_TRIANGLES);

            inline RenderTechnique* renderTechnique(const Core::Index& idx)
            { return m_renderTechniques.at(idx).get(); }
            inline const RenderTechnique* renderTechnique(const Core::Index& idx) const
            { return m_renderTechniques.at(idx).get(); }

            inline Material* material(const Core::Index& idx)
            { return m_materials.at(idx).get(); }
            inline const Material* material(const Core::Index& idx) const
            { return m_materials.at(idx).get(); }

            inline ShaderProgram* shaderProgram(const Core::Index& idx)
            { return m_shaderPrograms.at(idx).get(); }
            inline const ShaderProgram* shaderProgram(const Core::Index& idx) const
            { return m_shaderPrograms.at(idx).get(); }

            inline Texture* texture(const Core::Index& idx)
            { return m_textures.at(idx).get(); }
            inline const Texture* texture(const Core::Index& idx) const
            { return m_textures.at(idx).get(); }

            inline Mesh* mesh(const Core::Index& idx)
            { return m_meshes.at(idx).get(); }
            inline const Mesh* mesh(const Core::Index& idx) const
            { return m_meshes.at(idx).get(); }

            inline void removeRenderTechnique(const Core::Index& idx)
            { m_renderTechniques.remove(idx); }

            inline void removeMaterial(const Core::Index& idx)
            { m_materials.remove(idx); }

            inline void removeShaderProgram(const Core::Index& idx)
            { m_shaderPrograms.remove(idx); }

            inline void removeMesh(const Core::Index& idx)
            { m_meshes.remove(idx); }

            inline void removeTexture(const Core::Index& idx)
            { m_textures.remove(idx); }

        private:
            Core::IndexMap<RenderTechniquePtr>  m_renderTechniques;
            Core::IndexMap<MaterialPtr>         m_materials;
            Core::IndexMap<MeshPtr>             m_meshes;
            Core::IndexMap<ShaderProgramPtr>    m_shaderPrograms;
            Core::IndexMap<TexturePtr>          m_textures;
        };
    }
}

#endif // RADIUMENGINE_ASSETMANAGER_HPP
