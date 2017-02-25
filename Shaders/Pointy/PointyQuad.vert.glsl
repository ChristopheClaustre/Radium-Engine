#include "../Structs.glsl"

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
//layout (location = 2) in vec3 in_tangent;
//layout (location = 3) in vec3 in_bitangent;
//layout (location = 4) in vec3 in_texcoord;
layout (location = 5) in vec4 in_color;
//layout (location = 6) in vec4 in_weights
//layout (location = 7) in vec4 in_weightIdx
layout (location = 8) in float in_radius;

uniform Transform transform;

layout (location = 0) out vec3  out_position;
layout (location = 1) out vec3  out_normal;
layout (location = 2) out vec4  out_color;
layout (location = 3) out vec3  out_eye;
layout (location = 4) out float out_radius;

void main()
{
    mat4 mvp = transform.proj * transform.view * transform.model;
    gl_Position = mvp * vec4(in_position, 1.0);

    vec4 pos = transform.model * vec4(in_position, 1.0);
    pos /= pos.w;
    vec3 normal = mat3(transform.worldNormal) * in_normal;

    vec3 eye = -transform.view[3].xyz * mat3(transform.view);

    out_position  = vec3(pos);
    out_normal    = normal;
    out_color     = in_color;
    out_eye       = vec3(eye);
    out_radius    = in_radius;
}
