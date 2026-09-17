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
option(ASTROSIMGPU_OFFLOAD "Compile the astrocyte update as an OpenMP target region" OFF)
# Kokkos is an alternative to the OpenMP target path, not an addition to it.
# Both express the same loop over the same per-cell function; the point of
# having both is to measure what the abstraction costs.
option(ASTROSIMGPU_KOKKOS "Run the astrocyte update through Kokkos" OFF)
# Native CUDA, as the reference point the two portable routes are measured
# against. Also mutually exclusive with them.
option(ASTROSIMGPU_CUDA "Run the astrocyte update through native CUDA" OFF)
# neuron_update_kernel compiles to 122 registers/thread (measured with ncu),
# well above astro_update_kernel's 94, and is occupancy-limited as a result.
# 0 (default) leaves the register count to the compiler, unchanged from
# today. A positive value passes -maxrregcount=N to nvcc for that one file
# only, trading register budget for more resident warps per SM -- sweep
# this empirically (e.g. 64, 80, 96, 110) and check with ncu that Local
# Memory spill stays at zero; if it doesn't, that value is too aggressive.
set(ASTROSIMGPU_NEURON_MAXREGCOUNT 0 CACHE STRING
    "Cap registers/thread for neuron_update_kernel via nvcc -maxrregcount (0 = compiler default)")

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
  # Kokkos CUDA builds compile through nvcc or nvcc_wrapper, which reject some
  # of these and emit a great deal of noise for the rest.
  if(ASTROSIMGPU_KOKKOS OR ASTROSIMGPU_CUDA)
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
  if(ASTROSIMGPU_OFFLOAD OR ASTROSIMGPU_KOKKOS)
    message(FATAL_ERROR "The device backends are alternatives; enable exactly one")
  endif()
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 90 is Hopper, which is what a GH200 is. Override for other hardware.
    set(CMAKE_CUDA_ARCHITECTURES 90)
  endif()
  target_sources(astrosimgpu_core PRIVATE src/astrocyte_cuda.cu src/neuron_cuda.cu)
  if(ASTROSIMGPU_NEURON_MAXREGCOUNT GREATER 0)
    set_source_files_properties(src/neuron_cuda.cu PROPERTIES
        COMPILE_OPTIONS "--maxrregcount=${ASTROSIMGPU_NEURON_MAXREGCOUNT}")
    message(STATUS "neuron_update_kernel: capped at ${ASTROSIMGPU_NEURON_MAXREGCOUNT} registers/thread")
  endif()
  target_compile_definitions(astrosimgpu_core PUBLIC ASTROSIMGPU_CUDA)
  set_target_properties(astrosimgpu_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # network.cpp is compiled by the host compiler, not nvcc, but it needs
  # <nvtx3/...> for the phase-timing ranges nsys reads. enable_language(CUDA)
  # populates this variable with the toolkit's include path; NVTX v3 itself
  # is header-only, so nothing needs linking.
  target_include_directories(astrosimgpu_core PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  message(STATUS "Native CUDA backend, architecture ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if(ASTROSIMGPU_KOKKOS)
  if(ASTROSIMGPU_OFFLOAD OR ASTROSIMGPU_CUDA)
    message(FATAL_ERROR "The device backends are alternatives; enable exactly one")
  endif()
  find_package(Kokkos REQUIRED)
  target_compile_definitions(astrosimgpu_core PUBLIC ASTROSIMGPU_KOKKOS)
  target_link_libraries(astrosimgpu_core PUBLIC Kokkos::kokkos)
  # Which device this targets was decided when Kokkos was built, not here.
  message(STATUS "Kokkos devices: ${Kokkos_DEVICES}")
  message(STATUS "Kokkos arch:    ${Kokkos_ARCH}")
  if(Kokkos_ENABLE_CUDA)
    # A lambda passed to parallel_for becomes a __device__ lambda, which nvcc
    # only accepts with this enabled. Kokkos usually sets it; assert it rather
    # than discover it in a wall of template errors.
    kokkos_check(OPTIONS CUDA_LAMBDA)
  endif()
endif()

if(ASTROSIMGPU_OFFLOAD)
  target_compile_definitions(astrosimgpu_core PUBLIC ASTROSIMGPU_OFFLOAD)
  if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "ASTROSIMGPU_OFFLOAD needs OpenMP")
  endif()
  message(STATUS "OpenMP target offload enabled -- set the offload flags for "
                 "your compiler via CMAKE_CXX_FLAGS, e.g. -mp=gpu -gpu=cc90")
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
