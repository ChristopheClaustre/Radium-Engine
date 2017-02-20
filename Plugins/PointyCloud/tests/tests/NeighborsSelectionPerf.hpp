#ifndef POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONPERF_HPP
#define POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONPERF_HPP

#include "Test.hpp"

#include <misc/PointyCloudFactory.hpp>

#include <APSS/NeighborsSelection.hpp>
#include <APSS/NeighborsSelectionWithRegularGrid.hpp>
#include <APSS/RegularGrid.hpp>

#include <Core/Time/Timer.hpp>

#include <iostream>
#include <fstream>

struct OutFile {
    OutFile(const std::string& str) : name(str) {
        std::ofstream stream;
        stream.open(name, std::ofstream::trunc);
        stream.close();
    }

    inline void printLine(int i, float f) const {
        std::ofstream stream;
        stream.open(name, std::ofstream::app);
        stream << i << " " << f << "\n";
        stream.close();
    }

    inline void printLine(int i, float f1, float f2) const {
        std::ofstream stream;
        stream.open(name, std::ofstream::app);
        stream << i << " " << f1 << " " << f2 << "\n";
        stream.close();
    }

    std::string name;
};

using namespace PointyCloudPlugin;

namespace PointyCloudTests
{
    class NeighborsSelectionPerf : public Test
    {
        void run() override
        {
            // load from file
            std::string path = "../../models/bunny.ply"; // models/ in Radium root directory
            std::shared_ptr<PointyCloud> cloud = PointyCloudFactory::makeFromFile(path);
            size_t size = cloud->m_points.size();

            LOG(logINFO) << "Size = " << size;

            const int Nr = 10;
            const int NcellCount = 100;
            const int Nindices = 200;

            // define different influence radius
            float rmin = 0.1;
            float rmax = 5.0;
            std::array<float, Nr> rVec;
            for(uint k = 0; k<Nr; ++k)
                rVec[k] = (double)k/(Nr-1)*(rmax-rmin)+rmin;

            // define different cells count
            int cellMax = 1000000;
            int cellMin = 1;
            std::array<int, NcellCount> cellCountVec;
            for (int k = 0; k < NcellCount; ++k)
                 cellCountVec[k] = std::round(std::cbrt(((double)k/(NcellCount-1))*(cellMax-cellMin)+cellMin));

            // define different indices uniformly distributed
            std::array<int, Nindices> idxVec;
            size_t idxMax = size-1;
            for(uint k = 0; k<Nindices; ++k)
                idxVec[k] = ((double)k/(Nindices-1))*idxMax;

            // open file to print results
            OutFile buildTimeFile("../../models/timeBuild.txt");
            OutFile timeRefFile("../../models/timeRef.txt");
            OutFile timeGridFile("../../models/timeGrid.txt");

            NeighborsSelection selectorRef(cloud, 0.0);
            Ra::Core::Timer::TimePoint start;

            const int Nloop = Nr*NcellCount;
            int loop = 0;

            // for each cells count
            for(auto& cellCount : cellCountVec)
            {
                NeighborsSelectionWithRegularGrid selectorGrid(cloud, 0.0, cellCount);
                float buildTime = selectorGrid.grid()->getBuildTime();
                buildTimeFile.printLine(cellCount*cellCount*cellCount, buildTime);

                // for each radius
                for(auto& r : rVec)
                {
                    double progress = (double)(loop++)/Nloop*100;
                    if(((int)std::floor(progress)%10)==0) {
                        LOG(logINFO) << std::floor(progress) << "%";
                    }

                    selectorRef.setInfluenceRadius(r);
                    selectorGrid.setInfluenceRadius(r);

                    // select for each index with simple method
                    start = Ra::Core::Timer::Clock::now();
                    for(auto& idx : idxVec)
                        std::vector<int> resRef = selectorRef.getNeighbors(cloud->m_points[idx]);
                    float timeRef = Ra::Core::Timer::getIntervalMicro(start, Ra::Core::Timer::Clock::now());
                    timeRefFile.printLine(cellCount*cellCount*cellCount, r, timeRef/Nindices);

                    // select for each index with grid
                    start = Ra::Core::Timer::Clock::now();
                    for(auto& idx : idxVec)
                        std::vector<int> resGrid = selectorGrid.getNeighbors(cloud->m_points[idx]);
                    float timeGrid = Ra::Core::Timer::getIntervalMicro(start, Ra::Core::Timer::Clock::now());
                    timeGridFile.printLine(cellCount*cellCount*cellCount, r, timeGrid/Nindices);
                }
            }
            RA_UNIT_TEST(true,"ok");
        }
    };
    RA_TEST_CLASS(NeighborsSelectionPerf);
}

#endif //POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONPERF_HPP
