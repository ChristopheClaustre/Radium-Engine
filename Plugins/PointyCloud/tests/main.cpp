#include "TestManager.hpp"

#include "tests/DummyTest.hpp"
#include "tests/NeighborsSelectionTest.hpp"

int main(int argc, char *argv[])
{
    if (! PointyCloudTests::TestManager::getInstance())
        PointyCloudTests::TestManager::createInstance();

    return PointyCloudTests::TestManager::getInstance()->run();
}
