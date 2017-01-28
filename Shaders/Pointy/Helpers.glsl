
vec3 getKd()
{
    return in_color.xyz;
}

vec3 getKs()
{
    return vec3(1.0);
}

float getNs()
{
    return 1.0;
}

vec3 getNormal()
{
    return normalize(in_normal);
}
