#ifndef POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONTEST2_HPP
#define POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONTEST2_HPP

#include "Test.hpp"

#include <misc/PointyCloudFactory.hpp>
#include <APSS/NeighborsSelection/NeighborsSelection.hpp>
#include <APSS/NeighborsSelection/NeighborsSelectionWithRegularGrid.hpp>

#include <cstdlib>

using namespace PointyCloudPlugin;

namespace PointyCloudTests
{

    class NeighborsSelectionTest2 : public Test
    {
        void run() override
        {
            std::shared_ptr<PointyCloud> cloud = PointyCloudFactory::makeFromFile("../../models/bunny_8k.ply");
            size_t size = cloud->m_points.size();

            const double r = 1.0;

            NeighborsSelection selectorRef(cloud, r);
            NeighborsSelectionWithRegularGrid selectorTest(cloud, r);

            // number of selection test
            const int testCount = 100;
            int idxMax = size-1;
            std::array<int, testCount> idxVec;
            for(int itest = 0; itest<testCount; itest++)
                idxVec[itest] = ((double)itest/(testCount-1))*idxMax;

            bool sameNeighbors = true;

            for(int itest = 0; itest<testCount; itest++)
            {
                // point
                APoint p = cloud->m_points.at(idxVec.at(itest));

                std::vector<int> resRef  = selectorRef.getNeighbors(p);
                std::vector<int> resTest = selectorTest.getNeighbors(p);

                sameNeighbors &= (resRef.size()==resTest.size());

                for(auto& idxRef : resRef)
                    sameNeighbors &= (std::find(resTest.begin(), resTest.end(), idxRef)!=resTest.end());
            }
            RA_UNIT_TEST(sameNeighbors, "Wrong neighbors");
        }
    };
    RA_TEST_CLASS(NeighborsSelectionTest2);
}

#endif //POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONTEST2_HPP
