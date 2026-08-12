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
  target_compile_options(astrosimgpu_core PRIVATE -Wall -Wextra -Wpedantic)
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
