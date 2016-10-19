#include <Engine/Managers/AssetManager.hpp>

#include <Core/Containers/MakeShared.hpp>

#include <Engine/Renderer/RenderTechnique/RenderTechnique.hpp>
#include <Engine/Renderer/RenderTechnique/Material.hpp>

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

        RA_SINGLETON_IMPLEMENTATION(AssetManager);
    }
}
