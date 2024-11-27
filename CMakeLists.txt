cmake_minimum_required(VERSION 3.18)

project(MSz LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wno-unused-variable")


if(NOT CMAKE_C_COMPILER)
    message(STATUS "Using system default C compiler")
else()
    message(STATUS "Using custom C compiler: ${CMAKE_C_COMPILER}")
endif()

if(NOT CMAKE_CXX_COMPILER)
    message(STATUS "Using system default C++ compiler")
else()
    message(STATUS "Using custom C++ compiler: ${CMAKE_CXX_COMPILER}")
endif()


set(SOURCES
    main.cpp
    main.cu
)

add_executable(${PROJECT_NAME} ${SOURCES})


find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

set(CUDA_ARCHITECTURES 75)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


find_library(CUDART_LIBRARY cudart HINTS
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    PATH_SUFFIXES libcudart.so
)
if(CUDART_LIBRARY)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDART_LIBRARY})
endif()


set(SZ3_USE_BUNDLED_ZSTD ON CACHE BOOL "Force SZ3 to use bundled Zstd")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build SZ3 as a static library")
add_subdirectory(external/SZ3 EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME} PRIVATE SZ3)


set(ZFP_BUILD_SHARED OFF CACHE BOOL "Build ZFP as a static library")
add_subdirectory(external/zfp EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME} PRIVATE zfp)

