
cmake_minimum_required(VERSION 3.18)


project(HelloWorld LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


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
