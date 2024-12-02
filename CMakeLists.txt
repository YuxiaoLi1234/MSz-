cmake_minimum_required(VERSION 3.20)

project(MSz LANGUAGES CXX CUDA C)

include(GNUInstallDirs)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)
find_package(SZ3 REQUIRED)
find_package(ZFP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(ZSTD IMPORTED_TARGET libzstd)

add_executable(MSz
    main.cpp
    main.cu
)
target_compile_features(MSz PRIVATE cxx_std_17)
target_compile_options(MSz PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler -Wno-unused-variable>)
target_link_libraries(MSz PRIVATE PkgConfig::ZSTD zfp::zfp SZ3::SZ3 CUDA::cudart OpenMP::OpenMP_CXX)
set_target_properties(MSz PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS MSz)

