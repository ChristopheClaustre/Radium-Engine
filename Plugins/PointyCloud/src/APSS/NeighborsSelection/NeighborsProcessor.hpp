#ifndef POINTYCLOUDPLUGIN_NEIGHBORSPROCESSOR_HPP
#define POINTYCLOUDPLUGIN_NEIGHBORSPROCESSOR_HPP

#include <APSS/PointyCloud.hpp>

namespace PointyCloudPlugin
{

class NeighborsProcessor
{
public:
    NeighborsProcessor(){}
    virtual  ~NeighborsProcessor(){}
    virtual inline void operator()(int idx){}
};

}

#endif // POINTYCLOUDPLUGIN_NEIGHBORSPROCESSOR_HPP
