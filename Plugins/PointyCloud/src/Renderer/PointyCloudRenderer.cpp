 
#include <src/Renderer/PointyCloudRenderer.hpp>

#include <iostream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <Core/Log/Log.hpp>
#include <Core/Math/ColorPresets.hpp>
#include <Core/Containers/Algorithm.hpp>
#include <Core/Containers/MakeShared.hpp>

#include <Engine/RadiumEngine.hpp>
#include <Engine/Renderer/OpenGL/OpenGL.hpp>
#include <Engine/Renderer/OpenGL/FBO.hpp>
#include <Engine/Renderer/RenderTechnique/RenderTechnique.hpp>
#include <Engine/Renderer/RenderTechnique/Material.hpp>
#include <Engine/Renderer/RenderTechnique/ShaderProgramManager.hpp>
#include <Engine/Renderer/RenderTechnique/ShaderProgram.hpp>
#include <Engine/Renderer/RenderTechnique/RenderParameters.hpp>

//#define NO_TRANSPARENCY
namespace Ra
{
    namespace Engine
    {

        PointyCloudRenderer::PointyCloudRenderer( uint width, uint height )
            : Renderer(width, height)
        {
        }

        PointyCloudRenderer::~PointyCloudRenderer()
        {
            ShaderProgramManager::destroyInstance();
        }

        void PointyCloudRenderer::initializeInternal()
        {
            initShaders();

            DebugRender::createInstance();
            DebugRender::getInstance()->initialize();
        }

        void PointyCloudRenderer::initShaders()
        {
        }

        void PointyCloudRenderer::updateStepInternal( const RenderData& renderData )
        {
        }

        void PointyCloudRenderer::renderInternal( const RenderData& renderData )
        {
        }

        void PointyCloudRenderer::resizeInternal()
        {
        }

    }
} // namespace Ra
