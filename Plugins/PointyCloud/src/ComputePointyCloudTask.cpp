#include "ComputePointyCloudTask.hpp"

#include <PointyCloudComponent.hpp>

namespace PointyCloudPlugin
{

    ComputePointyCloudTask::ComputePointyCloudTask(PointyCloudComponent* component) :
        m_component(component)
    {
    }

    std::string ComputePointyCloudTask::getName() const
    {
        return "PointyCloudPlugin";
    }

    void ComputePointyCloudTask::process()
    {
        m_component->computePointyCloud();
    }
}
