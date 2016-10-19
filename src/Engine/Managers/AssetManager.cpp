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

        Core::Index AssetManager::createShaderProgram()
        {
            ShaderProgramPtr prog = Core::make_shared<ShaderProgram>();
            Core::Index result = m_shaderPrograms.insert(prog);

            CORE_ASSERT(result.isValid(), "Invalid index");

            return result;
        }

        Core::Index AssetManager::createTexture()
        {
            TexturePtr tex = Core::make_shared<Texture>();
            Core::Index result = m_textures.insert(tex);

            CORE_ASSERT(result.isValid(), "Invalid index");

            return result;
        }

        Core::Index AssetManager::createMesh(std::string name, GLenum renderMode)
        {
            MeshPtr mesh = Core::make_shared<Mesh>(name, renderMode);
            Core::Index result = m_meshes.insert(tex);

            CORE_ASSERT(result.isValid(), "Invalid index");

            return result;
        }

        RA_SINGLETON_IMPLEMENTATION(AssetManager);
    }
}
