#include "UsefulPointsSelection.hpp"

#include <algorithm>

namespace PointyCloudPlugin
{

UsefulPointsSelection::UsefulPointsSelection(std::shared_ptr<PointyCloud> originalCloud, const Ra::Engine::Camera* camera) :
    m_originalCloud(originalCloud),
    m_camera(camera)
{
}

UsefulPointsSelection::~UsefulPointsSelection()
{
}

PointyCloud UsefulPointsSelection::selectUsefulPoints()
{
    // check visibility from camera
    PointyCloud visiblePoints(*m_originalCloud.get());
    selectFromVisibility(visiblePoints);

    // this method must return at least 1 point
    if (visiblePoints.m_points.size() == 0)
        visiblePoints.m_points.push_back(m_originalCloud->m_points[0]);

    // check orientation
    PointyCloud wellOrientedPoints(visiblePoints);
    selectFromOrientation(wellOrientedPoints);

    // camera must be inside the cloud
    // if it is the case we must return the opposite face of the "mesh"
    //   (the not well oriented one (all the visible in fact))
    if (wellOrientedPoints.m_points.size() == 0)
        return visiblePoints;

    return wellOrientedPoints;
}

} // namespace PointyCloudPlugin












