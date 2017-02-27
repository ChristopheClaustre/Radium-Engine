#include "UsefulPointsSelection.hpp"

#include <algorithm>

namespace PointyCloudPlugin
{

UsefulPointsSelection::UsefulPointsSelection(std::shared_ptr<PointyCloud> originalCloud, const Ra::Engine::Camera* camera) :
    m_originalCloud(originalCloud), m_camera(camera)
{
    for(int i = 0; i < m_originalCloud->size(); ++i) {
        m_indices.push_back(i);
    }
}

UsefulPointsSelection::~UsefulPointsSelection()
{
}

void UsefulPointsSelection::selectUsefulPoints()
{
    // check visibility from camera
    selectFromVisibility();

    // this method must return at least 1 point
    if (m_N == 0 && m_indices.size() > 0) {
        m_N = 1;
    }
    int number_of_visible = m_indices.size();

    // check orientation
    selectFromOrientation();

    // camera may be inside the cloud
    // if it is the case we must return the opposite face of the "mesh"
    //   (the not well oriented one (all the visible in fact))
    if (m_N == 0 && m_indices.size() > number_of_visible) {
        m_N = number_of_visible;
    }
}

} // namespace PointyCloudPlugin
