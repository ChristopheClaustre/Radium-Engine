#ifndef POINTYCLOUDPLUGIN_DEFINES_H
#define POINTYCLOUDPLUGIN_DEFINES_H

#ifdef __CUDA_ARCH__
# define MULTIARCH __host__ __device__
#else
# define MULTIARCH
#endif

#endif // POINTYCLOUDPLUGIN_DEFINES_H
