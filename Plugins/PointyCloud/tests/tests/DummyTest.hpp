#ifndef POINTYCLOUDPLUGIN_DUMMYTEST_HPP
#define POINTYCLOUDPLUGIN_DUMMYTEST_HPP

#include "Test.hpp"

namespace PointyCloudTests
{
    class DummyTest : public Test
    {
        void run() override
        {
            RA_UNIT_TEST(42==42, "42 should be equal to 42 !");
        }
    };
    RA_TEST_CLASS(DummyTest);
}

#endif //POINTYCLOUDPLUGIN_DUMMYTEST_HPP
