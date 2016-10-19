#include <Engine/Managers/AssetManager.hpp>

#include <Core/Containers/MakeShared.hpp>

#include <Engine/Renderer/RenderTechnique/RenderTechnique.hpp>
#include <Engine/Renderer/RenderTechnique/Material.hpp>
#include <Engine/Renderer/RenderTechnique/ShaderProgram.hpp>
#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <Engine/Renderer/Texture/Texture.hpp>

namespace Ra
{
    namespace Engine
    {
        Core::Index AssetManager::createRenderTechnique()
        {
            RenderTechniquePtr rt = Core::make_shared<RenderTechnique>();
            Core::Index result = m_renderTechniques.insert(rt);

            CORE_ASSERT(result.isValid(), "Invalid index");

            return result;
        }

        Core::Index AssetManager::createMaterial(std::string name)
        {
            MaterialPtr mat = Core::make_shared<Material>(name);
            Core::Index result = m_materials.insert(mat);

            CORE_ASSERT(result.isValid(), "Invalid index");

            return result;
        }

        Core::Index AssetManager::createTexture(std::string name)
        {
            TexturePtr tex = Core::make_shared<Texture>(name);
            Core::Index result = m_textures.insert(tex);

            CORE_ASSERT(result.isValid(), "Invalid index");

            return result;
        }

        Core::Index AssetManager::createMesh(std::string name, GLenum renderMode)
        {
            MeshPtr mesh = Core::make_shared<Mesh>(name, renderMode);
            Core::Index result = m_meshes.insert(mesh);

            CORE_ASSERT(result.isValid(), "Invalid index");

            return result;
        }

        Core::Index AssetManager::createShaderProgram()
        {
            ShaderProgramPtr prog = Core::make_shared<ShaderProgram>();
            Core::Index result = m_shaderPrograms.insert(prog);

            CORE_ASSERT(result.isValid(), "Invalid index");

            return result;
        }

        Core::Index AssetManager::createShaderProgram(std::string vertFile,
                                                      std::string fragFile,
                                                      std::string geomFile,
                                                      std::string tescFile,
                                                      std::string teseFile)
        {
            ShaderConfiguration config;

            config.addShader(ShaderType_VERTEX, vertFile);
            config.addShader(ShaderType_FRAGMENT, fragFile);
            config.addShader(ShaderType_GEOMETRY, geomFile);
            config.addShader(ShaderType_TESS_CONTROL, tescFile);
            config.addShader(ShaderType_TESS_EVALUATION, teseFile);

            return createShaderProgram(config);
        }

        Core::Index AssetManager::createShaderProgram(std::string compFile)
        {
            ShaderConfiguration config;
            config.addShader(ShaderType_COMPUTE, compFile);
            return createShaderProgram(config);
        }

        Core::Index AssetManager::createShaderProgram(const ShaderConfiguration& config)
        {
            Core::Index result = createShaderProgram();
            ShaderProgram* prog = shaderProgram(result);

            prog->load(config);

            return result;
        }

        void AssetManager::reloadShaderPrograms()
        {
            for (auto prog : m_shaderPrograms)
            {
                prog->reload();
            }
        }

        RA_SINGLETON_IMPLEMENTATION(AssetManager);
    }
}
