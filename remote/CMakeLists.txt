set(PROJECT_REMOTE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(app)
add_subdirectory(graph)
add_subdirectory(serialization)
add_subdirectory(tools)
if (${WITH_CUDA})
    add_subdirectory(cuda)
endif()
