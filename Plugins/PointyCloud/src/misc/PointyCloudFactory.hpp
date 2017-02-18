#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDFACTORY_HPP
#define POINTYCLOUDPLUGIN_POINTYCLOUDFACTORY_HPP

#include <memory>

namespace PointyCloudPlugin {

class PointyCloud;

class PointyCloudFactory
{
public:

    static std::shared_ptr<PointyCloud> makeDenseCube(int n = 11, double dl = 0.1);

};

}

#endif
