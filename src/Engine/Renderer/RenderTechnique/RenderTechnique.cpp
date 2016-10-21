#include <Engine/Renderer/RenderTechnique/RenderTechnique.hpp>

#include <Engine/Managers/AssetManager.hpp>
#include <Engine/Renderer/RenderTechnique/Material.hpp>
#include <Engine/Renderer/RenderTechnique/ShaderProgram.hpp>

namespace Ra
{
    void Engine::RenderTechnique::changeShader(const ShaderConfiguration& newConfig)
    {
        shaderConfig  = newConfig;
        shaderIsDirty = true;
    }

    void Engine::RenderTechnique::updateGL()
    {
        if (nullptr == shader || shaderIsDirty)
        {
            AssetManager* mgr = AssetManager::getInstance();
            shader            = mgr->shaderProgram(mgr->createShaderProgram(shaderConfig));
            shaderIsDirty     = false;
        }

        if (material)
        {
            material->updateGL();
        }
    }

} // namespace Ra
