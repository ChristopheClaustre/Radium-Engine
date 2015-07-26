#ifndef FANCYMESHPLUGIN_FANCYMESHSYSTEM_HPP
#define FANCYMESHPLUGIN_FANCYMESHSYSTEM_HPP

#include <Engine/Entity/System.hpp>


namespace Ra { namespace Core { struct TriangleMesh; } }

namespace Ra { namespace Engine { class RadiumEngine;      } }
namespace Ra { namespace Engine { class Entity;            } }
namespace Ra { namespace Engine { class Component;         } }
namespace Ra { namespace Engine { class FancyMeshComponent;} }

namespace Ra { namespace Engine {

class FancyMeshSystem : public System
{
public:
	FancyMeshSystem(RadiumEngine* engine);
	virtual ~FancyMeshSystem();

	virtual void initialize() override;
	virtual void update(Scalar dt) override;
	virtual void handleFileLoading(const std::string& filename) override;

    virtual Component* addComponentToEntity( Engine::Entity* entity ) override;

    // Specialized factory methods for this systems.
    FancyMeshComponent* addDisplayMeshToEntity(Engine::Entity* entity, const Core::TriangleMesh& mesh);

private:
};

} // namespace Engine
} // namespace Ra

#endif // FANCYMESHPLUGIN_FANCYMESHSYSTEM_HPP
