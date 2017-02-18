#ifndef POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONTEST_HPP
#define POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONTEST_HPP

#include "Test.hpp"

#include <misc/PointyCloudFactory.hpp>
#include <APSS/NeighborsSelection.hpp>
#include <APSS/NeighborsSelectionWithRegularGrid.hpp>


using namespace PointyCloudPlugin;

namespace PointyCloudTests
{
    class NeighborsSelectionTest : public Test
    {
        void run() override
        {
            std::shared_ptr<PointyCloud> cloud = PointyCloudFactory::makeDenseCube(11, 0.1);

            NeighborsSelection selectorRef(cloud, 0.15);
            NeighborsSelectionWithRegularGrid selectorTest(cloud, 0.15);

            std::vector<int> resRef = selectorRef.getNeighbors(cloud->m_points.at(0).pos());
            std::vector<int> resTest = selectorTest.getNeighbors(cloud->m_points.at(0).pos());

            RA_UNIT_TEST(resRef.size()==resTest.size(), "Wrong neigbors count");
        }
    };
    RA_TEST_CLASS(NeighborsSelectionTest);
}

#endif //POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONTEST_HPP
