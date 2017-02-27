#ifndef POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONPERF_HPP
#define POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONPERF_HPP

#include "Test.hpp"

#include <misc/PointyCloudFactory.hpp>

#include <APSS/NeighborsSelection/NeighborsSelection.hpp>
#include <APSS/NeighborsSelection/NeighborsSelectionWithRegularGrid.hpp>
#include <APSS/UsefulPointsSelection.hpp>
#include <APSS/Projection/OrthogonalProjection.hpp>
#include <APSS/UpSampler/UpSamplerUnshaken.hpp>

#include <Core/Time/Timer.hpp>

#include <Engine/Renderer/Camera/Camera.hpp>
#include <Engine/Renderer/Mesh/Mesh.hpp>

#include <iostream>
#include <fstream>

using namespace PointyCloudPlugin;

namespace PointyCloudTests
{
    struct Stat
    {
    public:
        Stat(const std::string& nameStr = "unamed", const std::string& unitStr = "") :
            name(nameStr), unit(unitStr), sum(0.0), min(std::numeric_limits<double>::max()), max(std::numeric_limits<double>::min()), count(0) {}

        void update(double data) {
            sum += data;
            min = std::min(min, data);
            max = std::max(max, data);
            ++count;
        }

        double mean() const {
            return sum/count;
        }

        std::string name;
        std::string unit;

        double sum;
        double min;
        double max;
        int count;
    };

    std::ostream& operator<<(std::ostream& os, const Stat& s)
    {
        os << s.name << " statistics (over " << s.count << " measurements) :\n" <<
              "min  : " << s.min << " " << s.unit << "\n" <<
              "mean : " << s.mean() << " " << s.unit << "\n" <<
              "max  : " << s.max << " " << s.unit;
        return os;
    }

    class NeighborsSelectionPerf : public Test
    {
        void run() override
        {
            // load from file
            std::string path = "../../models/bunny_34k.ply"; // models/ in Radium root directory
            std::shared_ptr<PointyCloud> cloud = PointyCloudFactory::makeFromFile(path);
            size_t size = cloud->m_points.size();

            LOG(logINFO) << "Size = " << size;

            // camera
            Ra::Engine::Camera camera(600.0, 800.0);
            camera.setPosition(Ra::Core::Vector3(0.0, 3.716, 15.976));
            camera.setDirection(Ra::Core::Vector3(0, 0, -1));
            camera.setZFar(1000);

            // mesh
            Ra::Engine::Mesh mesh("test", GL_POINTS);

            // APSS parameters
            const int M = 3;
            const float r = 1.0;

            // create APSS stuff
            std::shared_ptr<NeighborsSelection> selector = std::make_shared<NeighborsSelection>(cloud, r);
            UsefulPointsSelection selection(*cloud.get(), &camera);
            UpSamplerUnshaken upsampler(M);
            OrthogonalProjection projection(selector, cloud, r);

            // test parameters
            const int testCount = 10;

            // stats
            Stat timeSelection("Selection time", "μs");
            Stat timeUpsampling("Upsampling time", "μs");
            Stat timeProjection("Projection time", "μs");
            Stat timeLoading("Loading time", "μs");
            Stat upsamplingCount("Upsampling count", "points");

            Ra::Core::Timer::TimePoint t0, t1, t2, t3, t4;
            int npoints;

            for (int iTest = 0; iTest < testCount; ++iTest)
            {
                LOG(logINFO) << std::round((double)iTest/(testCount-1)*100) << " %";

                t0 = Ra::Core::Timer::Clock::now();

                selection.selectUsefulPoints();
                PointyCloud& points = selection.getUsefulPoints();
                npoints = points.size();

                t1 = Ra::Core::Timer::Clock::now();

                upsampler.upSampleCloud(points, selection.getN());
                points = upsampler.getUpsampledCloud();

                t2 = Ra::Core::Timer::Clock::now();

                projection.project(points);

                t3 = Ra::Core::Timer::Clock::now();

                points.loadToMesh(&mesh);

                t4 = Ra::Core::Timer::Clock::now();

                timeSelection.update(Ra::Core::Timer::getIntervalMicro(t0,t1));
                timeUpsampling.update(Ra::Core::Timer::getIntervalMicro(t1,t2));
                timeProjection.update(Ra::Core::Timer::getIntervalMicro(t2,t3));
                timeLoading.update(Ra::Core::Timer::getIntervalMicro(t3,t4));
                upsamplingCount.update(npoints);
            }

            LOG(logINFO) << "Timing results :\n" <<
                timeSelection << "\n" <<
                timeUpsampling << "\n" <<
                timeProjection << "\n" <<
                timeLoading << "\n" <<
                upsamplingCount;

            RA_UNIT_TEST(true,"ok");
        }
    };
    RA_TEST_CLASS(NeighborsSelectionPerf);
}

#endif //POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONPERF_HPP
