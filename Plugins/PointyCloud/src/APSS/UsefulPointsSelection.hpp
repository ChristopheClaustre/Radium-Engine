#ifndef POINTYCLOUDPLUGIN_USEFULPOINTSSELECTION_HPP
#define POINTYCLOUDPLUGIN_USEFULPOINTSSELECTION_HPP

//#include <memory>
#include <vector>
#include <Engine/Renderer/Camera/Camera.hpp>

#include <APSS/PointyCloud.hpp>

namespace PointyCloudPlugin
{

class UsefulPointsSelection
{
public:
    UsefulPointsSelection(PointyCloud& originalCloud, const Ra::Engine::Camera* camera);
    ~UsefulPointsSelection();

    void selectUsefulPoints();
    inline PointyCloud& getUsefulPoints() { return m_cloud; }
    inline size_t getN() { return m_N; }

private:
    /// a cloud where the useful points are the this->N first points
    PointyCloud m_cloud;
    /// number of points to take care of, may be lower than m_cloud.size()
    size_t m_N;

    const Ra::Engine::Camera* m_camera;

    inline void selectFromVisibility();
    inline void selectFromOrientation();
}; // class UsefulPointsSelection

} // namespace PointyCloudPlugin

#include <APSS/UsefulPointsSelection.inl>

#endif // POINTYCLOUDPLUGIN_USEFULPOINTSSELECTION_HPP
