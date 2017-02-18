#ifndef POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONTEST_HPP
#define POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONTEST_HPP

#include "Test.hpp"

#include <misc/PointyCloudFactory.hpp>
#include <APSS/NeighborsSelection.hpp>
#include <APSS/NeighborsSelectionWithRegularGrid.hpp>

#include <cstdlib>

using namespace PointyCloudPlugin;

namespace PointyCloudTests
{

    inline float random(float a, float b) {
        return (float)std::rand()/RAND_MAX * (b-a) + a;
    }

    class NeighborsSelectionTest : public Test
    {
        void run() override
        {
            // points in [0,1]x[0,1]x[0,1]
            std::shared_ptr<PointyCloud> cloud = PointyCloudFactory::makeDenseCube(11, 0.1);

            NeighborsSelection selectorRef(cloud, 0.0);
            NeighborsSelectionWithRegularGrid selectorTest(cloud, 0.0);

            // number of selection test
            const int testCount = 25;

            std::srand(std::time(0));
            Ra::Core::Vector3 clamp3;
            Ra::Core::Vector4 clamp4;
            clamp3.fill(1.0);
            clamp4.fill(1.0);

            for(int itest = 0; itest<testCount; itest++)
            {
                // random radius
                float r = random(0.1, 1.0);
                selectorRef.setInfluenceRadius(r);
                selectorTest.setInfluenceRadius(r);

                // random point
                Ra::Core::Vector3 pos = 0.5*(Ra::Core::Vector3::Random()+clamp3);
                Ra::Core::Vector3 nor = Ra::Core::Vector3::Random().normalized();
                Ra::Core::Vector4 col = 0.5*(Ra::Core::Vector4::Random()+clamp4);

                APoint p(pos, nor, col);

                std::vector<int> resRef  = selectorRef.getNeighbors(p);
                std::vector<int> resTest = selectorTest.getNeighbors(p);

                RA_UNIT_TEST(resRef.size()==resTest.size(), "Wrong neigbors count");

                for(auto& idxRef : resRef)
                {
                    RA_UNIT_TEST(std::find(resTest.begin(), resTest.end(), idxRef)!=resTest.end(), "Wrong neighbors");
                }
            }
        }
    };
    RA_TEST_CLASS(NeighborsSelectionTest);
}

#endif //POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONTEST_HPP
