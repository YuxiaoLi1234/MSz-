cmake_minimum_required(VERSION 3.18)

project(MSz LANGUAGES CXX CUDA)

find_program(GCC_COMPILER gcc-12)
find_program(GPP_COMPILER g++-12)

if(NOT GCC_COMPILER OR NOT GPP_COMPILER)
    message(FATAL_ERROR "gcc-12 or g++-12 not found. Please ensure they are installed and available in PATH.")
else()
    set(CMAKE_C_COMPILER ${GCC_COMPILER} CACHE STRING "C Compiler" FORCE)
    set(CMAKE_CXX_COMPILER ${GPP_COMPILER} CACHE STRING "CXX Compiler" FORCE)
    message(STATUS "Using gcc-12: ${GCC_COMPILER}")
    message(STATUS "Using g++-12: ${GPP_COMPILER}")
endif()

set(CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


set(SOURCES
    main.cpp
    main.cu
)

add_executable(${PROJECT_NAME} ${SOURCES})

# OpenMP 支持
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()


set(SZ3_USE_BUNDLED_ZSTD ON CACHE BOOL "Use bundled Zstd") 
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build SZ3 as static library")
set(SZ3_BUILD_EXECUTABLE OFF CACHE BOOL "Do not build SZ3 CLI executable")
add_subdirectory(external/SZ3 EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME} PRIVATE SZ3)


set(ZFP_BUILD_SHARED OFF CACHE BOOL "Build ZFP as static library")
add_subdirectory(external/zfp EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME} PRIVATE zfp)


message(STATUS "Build configuration completed successfully.")
