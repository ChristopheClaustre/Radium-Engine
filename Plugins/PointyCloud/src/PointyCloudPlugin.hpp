#ifndef POINTYCLOUDPLUGIN_HPP_
#define POINTYCLOUDPLUGIN_HPP_

#include <Core/CoreMacros.hpp>
#include <QObject>
#include <QtPlugin>
#include <PluginBase/RadiumPluginInterface.hpp>

/// Defines the correct macro to export dll symbols.
#if defined  PointyCloud_EXPORTS
    #define POINTY_PLUGIN_API DLL_EXPORT
#else
    #define POINTY_PLUGIN_API DLL_IMPORT
#endif


namespace Ra
{
    namespace Engine
    {
        class RadiumEngine;
    }
}

namespace PointyCloudPlugin
{
    enum UPSAMPLING_METHOD {
        FIXED_METHOD = 0,
        SIMPLE_METHOD,
        COMPLEX_METHOD,
        MAX_UPSAMPLING_METHOD
    };
    enum PROJECTION_METHOD {
        ORTHOGONAL_METHOD = 0,
        ALMOST_METHOD,
        MAX_PROJECTION_METHOD
    };

    class PointyCloudPluginC : public QObject, Ra::Plugins::RadiumPluginInterface
    {
        Q_OBJECT
        Q_PLUGIN_METADATA( IID "RadiumEngine.PluginInterface" )
        Q_INTERFACES( Ra::Plugins::RadiumPluginInterface )

    public:
        PointyCloudPluginC();
        virtual ~PointyCloudPluginC();

        virtual void registerPlugin(const Ra::PluginContext& context) override;

        virtual bool doAddWidget( QString& name ) override;
        virtual QWidget* getWidget() override;

        virtual bool doAddMenu() override;
        virtual QMenu* getMenu() override;

        std::string static getName(UPSAMPLING_METHOD m) { return UPSAMPLING_METHOD_STR[m]; }
        std::string static getName(PROJECTION_METHOD m) { return PROJECTION_METHOD_STR[m]; }

    public slots:
        void setSplatRadius(float);
        void setInfluenceRadius(float);
        void setBeta(float);
        void setThreshold(float);
        void setUpsamplingMethod(UPSAMPLING_METHOD);
        void setProjectionMethod(PROJECTION_METHOD);
        void setOptimizationByOctree(bool);
        void setOptimizationByCUDA(bool);

    private:
        class PointyCloudSystem * m_system;

    public:
        static const std::array<std::string,MAX_UPSAMPLING_METHOD> UPSAMPLING_METHOD_STR;
        static const std::array<std::string,MAX_PROJECTION_METHOD> PROJECTION_METHOD_STR;
    };

} // namespace

#endif // POINTYCLOUDPLUGIN_HPP_
