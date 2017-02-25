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

#define LOGP(log) LOG(log) << "PointyCloudPlugin : "

#define TIMED
#ifndef TIMED
#   define ON_TIMED(CODE) /* rien */
#else
#   define ON_TIMED(CODE) CODE
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

    struct QDoubleSpinBoxInit {
        double min, max, step, init;
    };

    struct QSpinBoxInit {
        int min, max, step, init;
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
        void setSplatRadius(Scalar);
        void setInfluenceRadius(Scalar);
        void setThreshold(int);
        void setM(int);
        void setUpsamplingMethod(UPSAMPLING_METHOD);
        void setProjectionMethod(PROJECTION_METHOD);
        void setOptimizationByOctree(bool);
        void setOptimizationByCUDA(bool);
        void setAPSS(bool);
        void setRenderer(bool);

    private:
        class PointyCloudSystem * m_system;

    public:
        static const std::array<std::string,MAX_UPSAMPLING_METHOD> UPSAMPLING_METHOD_STR;
        static const std::array<std::string,MAX_PROJECTION_METHOD> PROJECTION_METHOD_STR;
        static constexpr QDoubleSpinBoxInit splatRadiusInit { 0.01,  5.0, 0.01, 0.5};
        static constexpr QDoubleSpinBoxInit influenceInit   { 0.01, 30.0, 0.01, 2.5};
        static constexpr QDoubleSpinBoxInit betaInit        {-8   ,  8  , 0.5 , 0.0};
        static constexpr QSpinBoxInit thresholdInit { 1,   5,  1, 1};
        static constexpr QSpinBoxInit mInit         { 1, 100, 2, 3};
    };

} // namespace

#endif // POINTYCLOUDPLUGIN_HPP_
