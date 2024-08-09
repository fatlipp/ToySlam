set(PROJ_NAME "graph_opt")
set(SOURCE
    main.cpp
    ../graph/EdgeSe2.cpp
    ../graph/EdgeSe2Point2d.cpp
)

find_package(Eigen3 REQUIRED)

add_executable(${PROJ_NAME})
target_sources(${PROJ_NAME}
    PRIVATE
        ${SOURCE}
)
target_include_directories(${PROJ_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_DIR}
)
target_link_libraries(${PROJ_NAME} 
    PRIVATE
    pthread
    boost_system
    boost_thread
    boost_serialization
    Eigen3::Eigen
)
set_property(TARGET ${PROJ_NAME}  PROPERTY POSITION_INDEPENDENT_CODE ON)