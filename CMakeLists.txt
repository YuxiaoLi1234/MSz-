cmake_minimum_required(VERSION 3.18)


project(MSz LANGUAGES CXX CUDA)


set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wno-unused-variable")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_COMPILER gcc-12)
set(CMAKE_CXX_COMPILER g++-12)

set(SOURCES
    main.cpp
    main.cu
)


add_executable(${PROJECT_NAME} ${SOURCES})


find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
    set(OpenMP_C_FOUND TRUE)
    set(OpenMP_C_LIBRARIES OpenMP::OpenMP_C)
    set(OpenMP_CXX_LIBRARIES OpenMP::OpenMP_CXX)
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

find_package(SZ3 REQUIRED)
if(SZ3_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${SZ3_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${SZ3_LIBRARIES})
else()
    message(FATAL_ERROR "SZ3 library not found.")
endif()

find_package(ZFP REQUIRED)
if(ZFP_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${ZFP_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ZFP_LIBRARIES})
else()
    message(FATAL_ERROR "ZFP library not found.")
endif()

find_package(zstd REQUIRED)
if(zstd_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${zstd_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${zstd_LIBRARIES})
else()
    message(FATAL_ERROR "zstd library not found.")
endif()
