#include "ComputePointyCloudTask.hpp"

namespace PointyCloudPlugin
{

    ComputePointyCloudTask::ComputePointyCloudTask(PointyCouldSystem* _system):system(_system)
    {
    }

    std::string ComputePointyCloudTask::getName() const
    {
        return "PointyCloudPlugin";
    }

    void ComputePointyCloudTask::process()
    {
        for(PointyCloudComponent component : system->getComponents())
        {
            component->computePointyCloud();
        }
    }
}
