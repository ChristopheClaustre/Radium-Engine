#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDTASK_HPP_
#define POINTYCLOUDPLUGIN_POINTYCLOUDTASK_HPP_

#include <Core/Tasks/Task.hpp>

namespace PointyCloudPlugin
{

    class PointyCloudComponent;

    class ComputePointyCloudTask: public Ra::Core::Task
    {
        public:
            ComputePointyCloudTask(PointyCloudComponent* component);
            std::string getName() const override;
            void process() override;

        private:
            PointyCloudComponent* m_component;
    };
}

#endif //POINTYCLOUDPLUGIN_POINTYCLOUDTASK_HPP_
