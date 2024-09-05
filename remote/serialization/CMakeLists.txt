set(PROJ_NAME "graph_serialization")
set(SOURCE
    SerializeHelper.cpp
)

find_package(Eigen3 REQUIRED)

add_library(${PROJ_NAME} STATIC)
target_sources(${PROJ_NAME}
    PRIVATE
        ${SOURCE}
)
target_include_directories(${PROJ_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_REMOTE_DIR}
)
target_link_libraries(${PROJ_NAME} 
    PRIVATE
        Eigen3::Eigen
)
set_property(TARGET ${PROJ_NAME}  PROPERTY POSITION_INDEPENDENT_CODE ON)