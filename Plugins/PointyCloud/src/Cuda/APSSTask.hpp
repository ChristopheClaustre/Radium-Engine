#ifndef POINTYCLOUDPLUGIN_APSSTASK_HPP
#define POINTYCLOUDPLUGIN_APSSTASK_HPP

#include <Core/Tasks/Task.hpp>

#include <Cuda/APSS.h>

#include <Engine/Renderer/Mesh/Mesh.hpp>
#include <Core/Math/LinearAlgebra.hpp>
#include <Engine/Renderer/Camera/Camera.hpp>

#include <Core/Time/Timer.hpp>

namespace PointyCloudPlugin {

    struct TimeStat {
        TimeStat() :
            timeSelection(0.0),
            timeUpsampling(0.0),
            timeProjection(0.0),
            timeFinalization(0.0),
            timeLoading(0.0),
            count(0) {}

        friend std::ostream& operator<<(std::ostream& os, const TimeStat& s);

        float timeSelection;
        float timeUpsampling;
        float timeProjection;
        float timeFinalization;
        float timeLoading;
        int   count;
    };

    std::ostream& operator<<(std::ostream& os, const TimeStat& s) {
        float total = s.timeSelection+s.timeUpsampling+s.timeProjection+s.timeFinalization+s.timeLoading;
        float totalMean = total/s.count;
        os << "Timing statistics in μs (for " << s.count <<" frames):\n" <<
            "selection    : " << s.timeSelection /s.count    << " (" << s.timeSelection/total*100 << "%)\n" <<
            "upsampling   : " << s.timeUpsampling /s.count   << " (" << s.timeUpsampling/total*100 << "%)\n" <<
            "projection   : " << s.timeProjection /s.count   << " (" << s.timeProjection/total*100 << "%)\n" <<
            "finalization : " << s.timeFinalization /s.count << " (" << s.timeFinalization/total*100 << "%)\n" <<
            "loading      : " << s.timeLoading /s.count      << " (" << s.timeLoading/total*100 << "%)\n" <<
            "total        : " << totalMean;
        return os;
    }

    class APSSTask : public Ra::Core::Task
    {
    public:

        APSSTask(Cuda::APSS* apss, std::shared_ptr<Ra::Engine::Mesh> mesh, TimeStat* stat, const Ra::Engine::Camera* camera, Scalar splatRadius, int M, Scalar influenceRadius) :
            m_apss(apss), m_mesh(mesh), m_stat(stat), m_camera(camera), m_splatRadius(splatRadius), m_M(M), m_influenceRadius(influenceRadius) {}
        ~APSSTask() {}

        virtual std::string getName() const override {return "APSS";}

        virtual void process() override
        {
            Ra::Core::Timer::Clock::time_point t0, t1, t2, t3, t4, t5;

            // APSS steps
            t0 = Ra::Core::Timer::Clock::now();
            m_apss->select(m_camera->getPosition(), m_camera->getDirection());
            t1 = Ra::Core::Timer::Clock::now();
            m_apss->upsample(m_M, m_splatRadius);
            t2 = Ra::Core::Timer::Clock::now();
            m_apss->project(m_influenceRadius);
            t3 = Ra::Core::Timer::Clock::now();
            m_apss->finalize();
            t4 = Ra::Core::Timer::Clock::now();

            // get results
            size_t size = m_apss->sizeFinal();
            const Ra::Core::Vector3* positions = m_apss->positionFinal();
            const Ra::Core::Vector3* normals = m_apss->normalFinal();
            const Ra::Core::Vector4* colors = m_apss->colorFinal();
            const Scalar* splatSizes = m_apss->splatSizeFinal();

            // send results to target mesh
            m_mesh->loadPointyCloud(size, positions, normals, colors, splatSizes);

            t5 = Ra::Core::Timer::Clock::now();

            ++m_stat->count;
            m_stat->timeSelection += Ra::Core::Timer::getIntervalMicro(t0, t1);
            m_stat->timeUpsampling += Ra::Core::Timer::getIntervalMicro(t1, t2);
            m_stat->timeProjection += Ra::Core::Timer::getIntervalMicro(t2, t3);
            m_stat->timeFinalization += Ra::Core::Timer::getIntervalMicro(t3, t4);
            m_stat->timeLoading += Ra::Core::Timer::getIntervalMicro(t4, t5);
        }

    private:

        // Cuda kernel calls
        Cuda::APSS* m_apss;

        // target mesh for rendering
        std::shared_ptr<Ra::Engine::Mesh> m_mesh;

        // timing statistics
        TimeStat* m_stat;

        // parameters
        const Ra::Engine::Camera* m_camera;
        Scalar                    m_splatRadius;
        int                       m_M;
        Scalar                    m_influenceRadius;
    };

} // namespace PointyCloudPlugin


#endif // POINTYCLOUDPLUGIN_APSSTASK_HPP
