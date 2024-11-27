cmake_minimum_required(VERSION 3.18)

project(MSz LANGUAGES CXX CUDA)

# 设置全局标准
set(CMAKE_C_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "C Compiler" FORCE)
set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "CXX Compiler" FORCE)

# 动态检测 g++-12 和 gcc-12
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

# 检测编译器版本
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} --version
    OUTPUT_VARIABLE GXX_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Detected CXX Compiler Version: ${GXX_VERSION}")
if(NOT GXX_VERSION MATCHES "12")
    message(FATAL_ERROR "g++-12 is required. Detected version: ${GXX_VERSION}.")
endif()

# 项目源文件
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

# CUDA 配置
set(CUDA_ARCHITECTURES 75)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 添加 SZ3
set(SZ3_USE_BUNDLED_ZSTD ON CACHE BOOL "Use bundled Zstd")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build SZ3 as static library")
add_subdirectory(external/SZ3 EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME} PRIVATE SZ3)

# 添加 ZFP
set(ZFP_BUILD_SHARED OFF CACHE BOOL "Build ZFP as static library")
add_subdirectory(external/zfp EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME} PRIVATE zfp)
