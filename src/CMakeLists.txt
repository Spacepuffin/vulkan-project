
# Add source to this project's executable.
add_executable(vulkan_project
    main.cpp
    vk_engine.cpp
    vk_engine.h
    vk_types.h
    vk_initializers.cpp
    vk_initializers.h
    vk_mesh.h
    vk_mesh.cpp)


set_property(TARGET vulkan_project PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:vulkan_project>")

target_include_directories(vulkan_project PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vulkan_project vkbootstrap vma glm tinyobjloader imgui stb_image)

target_link_libraries(vulkan_project Vulkan::Vulkan sdl2)

add_dependencies(vulkan_project Shaders)
