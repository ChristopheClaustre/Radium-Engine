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
    UsefulPointsSelection(std::shared_ptr<PointyCloud> originalCloud, const Ra::Engine::Camera* camera);
    ~UsefulPointsSelection();

    void selectUsefulPoints();
    inline const std::vector<unsigned int>& getIndices() const { return m_indices; }
    inline size_t getN() const { return m_N; }

private:
    /// a cloud where the useful points are the this->N first points
    std::shared_ptr<PointyCloud> m_originalCloud;
    std::vector<unsigned int> m_indices;
    /// number of points to take care of, may be lower than m_cloud.size()
    size_t m_N;

    const Ra::Engine::Camera* m_camera;

    inline void selectFromVisibility();
    inline void selectFromOrientation();
}; // class UsefulPointsSelection

} // namespace PointyCloudPlugin

#include <APSS/UsefulPointsSelection.inl>

#endif // POINTYCLOUDPLUGIN_USEFULPOINTSSELECTION_HPP
