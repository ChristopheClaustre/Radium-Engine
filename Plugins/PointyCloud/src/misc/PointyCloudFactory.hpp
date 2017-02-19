#ifndef POINTYCLOUDPLUGIN_POINTYCLOUDFACTORY_HPP
#define POINTYCLOUDPLUGIN_POINTYCLOUDFACTORY_HPP

#include <memory>

namespace PointyCloudPlugin {

class PointyCloud;

class PointyCloudFactory
{
public:

    static std::shared_ptr<PointyCloud> makeDenseCube(int n = 11, double dl = 0.1);

    static std::shared_ptr<PointyCloud> makeSphere(double radius = 1.0, int n = 20, int m = 10);

    static std::shared_ptr<PointyCloud> makeRandom(int n = 100, float xmin = 0.0, float xmax = 1.0,
                                                                float ymin = 0.0, float ymax = 1.0,
                                                                float zmin = 0.0, float zmax = 1.0);

};

}

#endif
