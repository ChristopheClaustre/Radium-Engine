#include "../Structs.glsl"

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout (location = 0) in vec3 in_position[];
layout (location = 1) in vec3 in_normal[];
layout (location = 2) in vec4 in_color[];
layout (location = 3) in vec3 in_eye[];

uniform float splatSize;
uniform Transform transform;

layout (location = 0) out vec3 out_position;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec4 out_color;
layout (location = 3) out vec3 out_eye;
layout (location = 4) out vec2 out_uv;

void main()
{
    // orthonormal basis {in_normal, u, v}
    vec3 u = mix(vec3(1,0,0), normalize(vec3(-in_normal[0].z/in_normal[0].x, 0, 1)), abs(in_normal[0].x)>0.0001);
    vec3 v = normalize(cross(in_normal[0], u));

    // quad corners and uv coordinates
    vec3 point[4];
    vec2 uv[4];

    point[0] = in_position[0]-0.5*splatSize*(u+v);
    uv[0] = vec2(-1,-1);

    point[1] = point[0] + splatSize*u;
    uv[1] = vec2(-1,+1);

    point[2] = point[0] + splatSize*v;
    uv[2] = vec2(+1,-1);

    point[3] = point[0] + splatSize*(u+v);
    uv[3] = vec2(+1,+1);

    for(int idx = 0; idx<4; ++idx)
    {
        gl_Position  = transform.proj * transform.view * vec4(point[idx],1);
        out_position = point[idx];
        out_normal   = in_normal[0];
        out_color    = in_color[0];
        out_eye      = in_eye[0];
        out_uv       = uv[idx];
        EmitVertex();
    }

    EndPrimitive();

}












