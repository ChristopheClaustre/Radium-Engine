#ifndef POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONTEST_HPP
#define POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONTEST_HPP

#include "Test.hpp"

namespace PointyCloudTests
{
    class NeighborsSelectionTest : public Test
    {
        void run() override
        {
            //TODO compare two selection method
            RA_UNIT_TEST(true, "ok");
        }
    };
    RA_TEST_CLASS(NeighborsSelectionTest);
}

#endif //POINTYCLOUDPLUGIN_NEIGHBORSSELECTIONTEST_HPP
