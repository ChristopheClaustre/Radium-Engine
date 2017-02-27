#include "UsefulPointsSelection.hpp"

#include <algorithm>

namespace PointyCloudPlugin
{

UsefulPointsSelection::UsefulPointsSelection(PointyCloud& originalCloud, const Ra::Engine::Camera* camera) :
    m_cloud(originalCloud), m_camera(camera)
{
}

UsefulPointsSelection::~UsefulPointsSelection()
{
}

void UsefulPointsSelection::selectUsefulPoints()
{
    // check visibility from camera
    selectFromVisibility();

    // this method must return at least 1 point
    if (m_N == 0 && m_cloud.size() > 0) {
        m_N = 1;
    }
    int number_of_visible = m_cloud.size();

    // check orientation
    selectFromOrientation();

    // camera may be inside the cloud
    // if it is the case we must return the opposite face of the "mesh"
    //   (the not well oriented one (all the visible in fact))
    if (m_N == 0 && m_cloud.size() > number_of_visible) {
        m_N = number_of_visible;
    }
}

} // namespace PointyCloudPlugin
