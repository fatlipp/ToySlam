find_package(Eigen3 REQUIRED)

set(CUDA_LIBS "")
set(CUDA_INCLUDE "")

if (${WITH_CUDA})
    get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(CUDA_DIR "${CUDA_COMPILER_BIN}" DIRECTORY)
    set(CUDA_INCLUDE "${CUDA_DIR}/include")

    set(CUDA_LIBS
            cublas
            cudart
            cusolver
            cuda_core
        )
endif()

message(STATUS "WITH_CUDA = ${WITH_CUDA}")
message(STATUS "CUDA_LIBS = ${CUDA_LIBS}")
message(STATUS "CUDA_INCLUDE = ${CUDA_INCLUDE}")

set(CUDA_PROJ "graph_optimizer")
set(CUDA_PROJ_SOURCE
    main.cpp
)

add_executable(${CUDA_PROJ})
target_sources(${CUDA_PROJ}
    PRIVATE
        ${CUDA_PROJ_SOURCE}
)
target_include_directories(${CUDA_PROJ}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_REMOTE_DIR}
        ${CUDA_INCLUDE}
)
target_link_libraries(${CUDA_PROJ} 
    PRIVATE
        boost_system
        boost_thread
        boost_serialization
        Eigen3::Eigen
        graph_core
        graph_serialization
        tools
        ${CUDA_LIBS}
)
set_property(TARGET ${CUDA_PROJ}  PROPERTY POSITION_INDEPENDENT_CODE ON)
target_precompile_headers(${CUDA_PROJ} PRIVATE pch.h)

if (${WITH_CUDA})
    target_compile_definitions(${CUDA_PROJ}  PRIVATE WITH_CUDA)

    target_compile_options(${CUDA_PROJ} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    set_target_properties(${CUDA_PROJ} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${CUDA_PROJ} PROPERTIES CUDA_ARCHITECTURES "86")
endif()