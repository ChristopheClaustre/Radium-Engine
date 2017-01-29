set( subdir "PointyCloud" )

set(libs
    radiumCore
    radiumEngine
    ${GLEW_LIBRARIES}
    ${Assimp_LIBRARIES}
    ${FREEIMAGE_LIBRARY}
    ${Qt5OpenGL_LIBRARIES}
    ${OPENGL_LIBRARIES}
)

file(GLOB_RECURSE sources *.cpp)
file(GLOB_RECURSE headers *.h *.hpp)
file(GLOB_RECURSE inlines *.inl)

file(GLOB_RECURSE app_uis *.ui)
qt5_wrap_ui(app_uis_moc ${app_uis})

add_library(
    ${subdir} SHARED
    ${sources}
    ${headers}
    ${inlines}
    ${app_uis})

target_compile_definitions(${subdir} PRIVATE "-D${subdir}_EXPORTS")

target_link_libraries(
    ${subdir}
    ${libs})
