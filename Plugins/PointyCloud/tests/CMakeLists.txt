set(target pointycloudtests)

file(GLOB_RECURSE sources *.cpp)
file(GLOB_RECURSE headers *.hpp)
file(GLOB_RECURSE inlines *.inl)

add_executable(
 ${target}
 ${sources}
 ${headers}
 ${inlines}
)

target_link_libraries(
 ${target}
 PointyCloud
)


