cmake_minimum_required(VERSION 4.2)

cmake_policy(SET CMP0168 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# build options
set(CACHE{RA_BUILD_TYPE} TYPE STRING VALUE "Release")
set(CACHE{RA_CCCL_TAG} TYPE STRING
  VALUE "09094af138841ef521de1adbbdd18ab8b3dad47b") # v3.3.0

# test options
set(CACHE{RA_USE_TEST} TYPE BOOL VALUE OFF)
set(CACHE{RA_TEST_MPI_SIZE} TYPE STRING VALUE 2)

# benchmark options
set(CACHE{RA_BENCHMARK_MPI_SIZE} TYPE STRING VALUE 2)
set(CACHE{RA_BENCHMARK_SAMPLE_SIZE} TYPE STRING VALUE 4)

list(APPEND _build_type "Release" "Debug")
if(CMAKE_BUILD_TYPE)
  if(${CMAKE_BUILD_TYPE} IN_LIST _build_type)
    set(RA_BUILD_TYPE ${CMAKE_BUILD_TYPE})
  endif()
endif()

project(ra VERSION 0.1.0 LANGUAGES CXX CUDA)
add_library(ra STATIC)

if(RA_BUILD_TYPE STREQUAL "Release")
  target_compile_options(ra PUBLIC -O3)
elseif(RA_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(
    ra
    PUBLIC
    -Wall
    -Wextra
    -g
    -O0
    -fno-omit-frame-pointer
    $<$<COMPILE_LANGUAGE:CUDA>:-G>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler>
    $<$<COMPILE_LANGUAGE:CUDA>:-rdynamic>
  )
  target_compile_definitions(
    ra
    PUBLIC
    -DCUB_STDERR
    -DRA_DEBUG
    <$<COMPILE_LANGUAGE:CUDA>:-DCUB_STDERR>
    <$<COMPILE_LANGUAGE:CUDA>:-DRA_DEBUG>
  )
endif()
target_compile_options(
  ra
  PUBLIC
  -fno-exceptions
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

find_package(CUDAToolkit 12.0 REQUIRED)
message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
message(STATUS "CUDAToolkit_LIBRARY_ROOT: ${CUDAToolkit_LIBRARY_ROOT}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(
  ra
  PUBLIC
  CUDA::cudart
  CUDA::cuda_driver
  CUDA::cublas
  CUDA::cublasLt
  CUDA::cusparse
  CUDA::cusolver
  CUDA::curand
)

find_package(MPI 3.0 REQUIRED COMPONENTS CXX)
target_link_libraries(ra PUBLIC MPI::MPI_CXX)

find_package(Git REQUIRED)
find_package(PkgConfig 2.0 REQUIRED)

pkg_check_modules(NetCDF REQUIRED IMPORTED_TARGET netcdf>=4.0)
target_link_libraries(ra PUBLIC PkgConfig::NetCDF)

include(FetchContent)

if(IS_READABLE "${PROJECT_SOURCE_DIR}/patch/cccl.openmp.${RA_CCCL_TAG}.patch")
  FetchContent_Declare(
    CCCL
    GIT_REPOSITORY "https://github.com/NVIDIA/cccl.git"
    GIT_TAG "${RA_CCCL_TAG}"
    PATCH_COMMAND ${GIT_EXECUTABLE} checkout .
      COMMAND ${GIT_EXECUTABLE} clean -fdx
      COMMAND ${GIT_EXECUTABLE} apply
        "${PROJECT_SOURCE_DIR}/patch/cccl.openmp.${RA_CCCL_TAG}.patch"
  )
else()
  FetchContent_Declare(
    CCCL
    GIT_REPOSITORY "https://github.com/NVIDIA/cccl.git"
    GIT_TAG "${RA_CCCL_TAG}"
  )
endif()
set(CACHE{CCCL_THRUST_HOST_SYSTEM} TYPE INTERNAL VALUE "OMP")
set(CACHE{CCCL_THRUST_DEVICE_SYSTEM} TYPE INTERNAL VALUE "CUDA")
FetchContent_MakeAvailable(CCCL)
target_link_libraries(ra PUBLIC CCCL::CCCL)

target_compile_features(ra PUBLIC cxx_std_20 cuda_std_20)
target_include_directories(ra PUBLIC include)
add_subdirectory(src)
add_subdirectory(example)

if(RA_USE_TEST)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG 29c9844f688acb27c87338c39cd186ebfe41aa19 # v3.13.0
  )
  FetchContent_MakeAvailable(Catch2)

  enable_testing()
  add_subdirectory(test)
  add_subdirectory(benchmark)
endif()
