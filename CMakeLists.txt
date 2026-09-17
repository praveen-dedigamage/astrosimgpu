cmake_minimum_required(VERSION 3.16)
project(astrosimgpu VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ASTROSIMGPU_NATIVE "Tune for the building machine (-march=native)" OFF)
option(ASTROSIMGPU_OPENMP "Use OpenMP for the per-cell update loops" ON)
option(ASTROSIMGPU_BUILD_TESTS "Build the test binary" ON)
# Off by default: the default build contains no device code.
option(ASTROSIMGPU_CUDA "Run the astrocyte update through native CUDA" OFF)

add_library(astrosimgpu_core
  src/astrocyte.cpp
  src/neuron.cpp
  src/network.cpp
  src/recorder.cpp
  src/analysis.cpp
  src/parameters.cpp
  src/json.cpp
)

target_include_directories(astrosimgpu_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(MSVC)
  target_compile_options(astrosimgpu_core PRIVATE /W4)
else()
  # nvcc/nvcc_wrapper reject some of these and emit a great deal of noise
  # for the rest.
  if(ASTROSIMGPU_CUDA)
    target_compile_options(astrosimgpu_core PRIVATE -Wall)
  else()
    target_compile_options(astrosimgpu_core PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  if(ASTROSIMGPU_NATIVE)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" HAVE_MARCH_NATIVE)
    if(HAVE_MARCH_NATIVE)
      target_compile_options(astrosimgpu_core PUBLIC -march=native)
    endif()
  endif()
endif()

if(ASTROSIMGPU_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(astrosimgpu_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled")
  else()
    message(STATUS "OpenMP not found; building single-threaded")
  endif()
endif()

if(ASTROSIMGPU_CUDA)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 90 is Hopper, which is what a GH200 is. Override for other hardware.
    set(CMAKE_CUDA_ARCHITECTURES 90)
  endif()
  # All device kernels (astrocyte, neuron, and the Stage 4 delivery
  # pipeline) live in this one translation unit -- see the file header in
  # cuda_kernels.cu for why they were consolidated.
  target_sources(astrosimgpu_core PRIVATE src/cuda_kernels.cu)
  target_compile_definitions(astrosimgpu_core PUBLIC ASTROSIMGPU_CUDA)
  set_target_properties(astrosimgpu_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # network.cpp is compiled by the host compiler, not nvcc, but it needs
  # <nvtx3/...> for the phase-timing ranges nsys reads. enable_language(CUDA)
  # populates this variable with the toolkit's include path; NVTX v3 itself
  # is header-only, so nothing needs linking.
  target_include_directories(astrosimgpu_core PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  message(STATUS "Native CUDA backend, architecture ${CMAKE_CUDA_ARCHITECTURES}")
endif()

add_executable(astrosimgpu src/main.cpp)
target_link_libraries(astrosimgpu PRIVATE astrosimgpu_core)

if(ASTROSIMGPU_BUILD_TESTS)
  enable_testing()
  add_executable(astrosimgpu_tests tests/test_astrosimgpu.cpp)
  target_link_libraries(astrosimgpu_tests PRIVATE astrosimgpu_core)
  add_test(NAME astrosimgpu_tests COMMAND astrosimgpu_tests)
endif()

install(TARGETS astrosimgpu RUNTIME DESTINATION bin)
