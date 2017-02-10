#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDTASK_HPP_
#define POINTYCLOUDPLUGIN_POINTYCLOUDTASK_HPP_

#include <Core/Tasks/Task.hpp>
#include "PointyCloudComponent.hpp"
#include "PointyCloudSystem.hpp"


namespace PointyCloudPlugin
{

    class ComputePointyCloudTask: public Ra::Core::Task
    {
        public:
            ComputePointyCloudTask(PointyCloudSystem* system);
            std::string getName() const override;
            void process() override;

        private:
            PointyCloudSystem* system;
    };
}

#endif //POINTYCLOUDPLUGIN_POINTYCLOUDTASK_HPP_
