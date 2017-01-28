#include "../Structs.glsl"

uniform Light light;

out vec4 fragColor;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec4 in_color;
layout (location = 3) in vec3 in_eye;

#include "../LightingFunctions.glsl"

void main()
{
    fragColor = vec4(computeLighting(), 1.0);
}
