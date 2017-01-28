#include <PointyCloudPlugin.hpp>

#include <Engine/RadiumEngine.hpp>

#include <PointyCloudSystem.hpp>

namespace PointyCloudPlugin
{

    PointyCloudPluginC::~PointyCloudPluginC()
    {
    }

    void PointyCloudPluginC::registerPlugin( const Ra::PluginContext& context )
    {
        PointyCloudSystem* system = new PointyCloudSystem;
        context.m_engine->registerSystem( "PointyCloudSystem", system );
    }

    bool PointyCloudPluginC::doAddWidget( QString &name )
    {
        return false;
    }

    QWidget* PointyCloudPluginC::getWidget()
    {
        return nullptr;
    }

    bool PointyCloudPluginC::doAddMenu()
    {
        return false;
    }

    QMenu* PointyCloudPluginC::getMenu()
    {
        return nullptr;
    }
}
