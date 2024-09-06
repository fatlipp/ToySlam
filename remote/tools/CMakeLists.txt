set(PROJ_NAME "tools")
set(SOURCE
    BlockTimer.cpp
)

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
set_property(TARGET ${PROJ_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)