 
#ifndef RADIUMENGINE_PLUGIN_POINTYCLOUDRENDERER_HPP
#define RADIUMENGINE_PLUGIN_POINTYCLOUDRENDERER_HPP

#include <Engine/RadiumEngine.hpp>
#include <Engine/Renderer/Renderer.hpp>

namespace Ra
{
    namespace Engine
    {
        class RA_ENGINE_API PointyCloudRenderer : public Renderer
        {
        public:
            PointyCloudRenderer( uint width, uint height );
            virtual ~PointyCloudRenderer();
            virtual std::string getRendererName() const override { return "PointyCloud Renderer"; }

            void setSplatSize(float size);
            float getSplatSize();

        protected:

            virtual void initializeInternal() override;
            virtual void resizeInternal() override;
            virtual void renderInternal( const RenderData& renderData ) override;

        private:
            void initShaders();


        private:
            float splatSize;

        };

    } // namespace Engine
} // namespace Ra

#endif // RADIUMENGINE_FORWARDRENDERER_HPP
