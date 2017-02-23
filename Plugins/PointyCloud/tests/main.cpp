#include "TestManager.hpp"

#include "tests/DummyTest.hpp"
#include "tests/NeighborsSelectionTest.hpp"
#include "tests/PerformanceAPSS.hpp"

//#include "tests/NeighborsSelectionPerf.hpp" <- be careful (could be long!)

int main(int argc, char *argv[])
{
    if (! PointyCloudTests::TestManager::getInstance())
        PointyCloudTests::TestManager::createInstance();

    return PointyCloudTests::TestManager::getInstance()->run();
}
