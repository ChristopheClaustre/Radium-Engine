#include <PointyCloudPlugin.hpp>

#include <QAction>
#include <QIcon>
#include <QToolBar>

#include <Engine/RadiumEngine.hpp>
#include <PointyCloudSystem.hpp>

#include <UI/PointyCloudUI.h>

namespace PointyCloudPlugin
{
    const std::array<std::string,MAX_UPSAMPLING_METHOD> PointyCloudPluginC::UPSAMPLING_METHOD_STR = {{ "Fixed", "Simple", "Complex" }};
    const std::array<std::string,MAX_PROJECTION_METHOD> PointyCloudPluginC::PROJECTION_METHOD_STR = {{ "Orthogonal", "A-Orthogonal" }};

    PointyCloudPluginC::PointyCloudPluginC() :m_system(nullptr){}

    PointyCloudPluginC::~PointyCloudPluginC()
    {
    }

    void PointyCloudPluginC::registerPlugin(const Ra::PluginContext& context)
    {
        m_system = new PointyCloudSystem;
        context.m_engine->registerSystem( "PointyCloudSystem", m_system );
    }

    bool PointyCloudPluginC::doAddWidget( QString &name )
    {
        name = "PointyCloud";
        return true;
    }

    QWidget* PointyCloudPluginC::getWidget()
    {
        PointyCloudUI* widget = new PointyCloudUI(
                    m_system->getSplatRadius(), m_system->getInfluenceRadius(), m_system->getBeta(),
                    m_system->getThreshold(), m_system->getUpsamplingMethod(), m_system->getProjectionMethod(),
                    m_system->isOptimizedByOctree(), m_system->isOptimizedByCUDA());

        connect( widget, &PointyCloudUI::setSplatRadius,        this, &PointyCloudPluginC::setSplatRadius );
        connect( widget, &PointyCloudUI::setInfluenceRadius,    this, &PointyCloudPluginC::setInfluenceRadius );
        connect( widget, &PointyCloudUI::setBeta,               this, &PointyCloudPluginC::setBeta );
        connect( widget, &PointyCloudUI::setThreshold,          this, &PointyCloudPluginC::setThreshold );
        connect( widget, &PointyCloudUI::setUpsamplingMethod,   this, &PointyCloudPluginC::setUpsamplingMethod );
        connect( widget, &PointyCloudUI::setProjectionMethod,   this, &PointyCloudPluginC::setProjectionMethod );
        connect( widget, &PointyCloudUI::setOptimizationByOctree,   this, &PointyCloudPluginC::setOptimizationByOctree );
        connect( widget, &PointyCloudUI::setOptimizationByCUDA,     this, &PointyCloudPluginC::setOptimizationByCUDA );

        return widget;
    }

    bool PointyCloudPluginC::doAddMenu()
    {
        return false;
    }

    QMenu* PointyCloudPluginC::getMenu()
    {
        return nullptr;
    }

    void PointyCloudPluginC::setSplatRadius(float splatRadius)
    {
        CORE_ASSERT(m_system, "System should be there ");
        m_system->setSplatRadius(splatRadius);
    }

    void PointyCloudPluginC::setInfluenceRadius(float influenceRadius)
    {
        CORE_ASSERT(m_system, "System should be there ");
        m_system->setInfluenceRadius(influenceRadius);
    }

    void PointyCloudPluginC::setBeta(float beta)
    {
        CORE_ASSERT(m_system, "System should be there ");
        m_system->setBeta(beta);
    }

    void PointyCloudPluginC::setThreshold(float threshold)
    {
        CORE_ASSERT(m_system, "System should be there ");
        m_system->setSplatRadius(threshold);
    }

    void PointyCloudPluginC::setUpsamplingMethod(UPSAMPLING_METHOD method)
    {
        CORE_ASSERT(m_system, "System should be there ");
        m_system->setUpsamplingMethod(method);
    }

    void PointyCloudPluginC::setProjectionMethod(PROJECTION_METHOD method)
    {
        CORE_ASSERT(m_system, "System should be there ");
        m_system->setProjectionMethod(method);
    }

    void PointyCloudPluginC::setOptimizationByOctree(bool octree)
    {
        CORE_ASSERT(m_system, "System should be there ");
        m_system->setOptimizationByOctree(octree);
    }

    void PointyCloudPluginC::setOptimizationByCUDA(bool cuda)
    {
        CORE_ASSERT(m_system, "System should be there ");
        m_system->setOptimizationByCUDA(cuda);
    }

}
