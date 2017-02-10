#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDTASK_HPP_
#define POINTYCLOUDPLUGIN_POINTYCLOUDTASK_HPP_

#include <Core/Tasks/Task.hpp>

namespace PointyCloudPlugin
{

    class ComputePointyCloudTask: public Ra::Core::Task
    {
        public:
            ComputePointyCloudTask(PointyCouldSystem* system);
            std::string getName() const override;
            void process() override;

        private:
            PointyCouldSystem* system;
    };
}

#endif //POINTYCLOUDPLUGIN_POINTYCLOUDTASK_HPP_
