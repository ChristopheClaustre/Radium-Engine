#include "../Structs.glsl"

uniform Light light;

out vec4 fragColor;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec4 in_color;
layout (location = 3) in vec3 in_eye;
layout (location = 4) in vec2 in_uv;

#include "../LightingFunctions.glsl"

void main()
{
    if(length(in_uv)>1.0)
        discard;
    else
        fragColor = vec4(computeLighting(),1);
}
