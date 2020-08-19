## CMake settings
# V3.9 required for first-class CUDA support
# V3.17 required for the FindCUDAToolkit package
# V3.18 required for CMAKE_CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.18)
find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)

## Project settings
project(astaroth C CXX CUDA)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## Project-wide compilation flags
set(COMMON_FLAGS "-mavx -Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

## CUDA
# GPU, compute capability
# K40, 3.5
# K80, 3.7
# P100, 6.0
# V100, 7.0
if (NOT CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70) # Default
else ()
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}) # User-specified
endif()
string (REPLACE " " "," CUDA_COMMON_FLAGS "${COMMON_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options=${CUDA_COMMON_FLAGS}")

## Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release") # Default
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

## Options
option(DOUBLE_PRECISION "Generates double precision code."                    OFF)
option(BUILD_SAMPLES    "Builds projects in samples subdirectory."            ON)
option(MPI_ENABLED      "Enables additional functions for MPI communciation." OFF)
option(MULTIGPU_ENABLED "Enables multi-GPU on a single node. Uses peer-to-peer communication instead of MPI. Affects Legacy & Node layers only." ON)

## Options (DEPRECATED)
# option(BUILD_DEBUG              "Builds the program with extensive error checking"          OFF)
# option(BUILD_UTILS              "Builds the utility library"                                ON)
# option(BUILD_RT_VISUALIZATION   "Builds the module for real-time visualization using SDL2"  OFF)


## Compile ACC
add_subdirectory(acc)

## Compile AC headers
if (NOT DEFINED DSL_MODULE_DIR)
    set(DSL_MODULE_DIR ${CMAKE_SOURCE_DIR}/acc/mhd_solver) # Default
endif ()
get_filename_component(DSL_MODULE_DIR ${DSL_MODULE_DIR} REALPATH)
message(STATUS "AC module dir: ${DSL_MODULE_DIR}")

file(GLOB DSL_SOURCES ${DSL_MODULE_DIR}/*
                      ${CMAKE_SOURCE_DIR}/acc/stdlib/*)
set(DSL_HEADERS "${PROJECT_BINARY_DIR}/user_kernels.h"
                "${PROJECT_BINARY_DIR}/user_defines.h"
                "${PROJECT_BINARY_DIR}/astaroth.f90")

add_custom_command (
    COMMENT "Building ACC objects ${DSL_MODULE_DIR}"
    COMMAND ${CMAKE_SOURCE_DIR}/acc/compile_acc_module.sh ${DSL_MODULE_DIR}
    DEPENDS ${DSL_SOURCES} acc
    OUTPUT ${DSL_HEADERS}
)
add_custom_target(dsl_headers ALL DEPENDS ${DSL_HEADERS})

## Global flags
if (DOUBLE_PRECISION)
    add_definitions(-DAC_DOUBLE_PRECISION=1)
endif ()
if (MPI_ENABLED)
    add_definitions(-DAC_MPI_ENABLED=1)
endif ()
if (MULTIGPU_ENABLED)
  add_definitions(-DAC_MULTIGPU_ENABLED=1)
endif()

add_definitions(-DAC_DEFAULT_CONFIG="${CMAKE_SOURCE_DIR}/config/astaroth.conf")

## Includes
include_directories(include)                                   # Library headers
include_directories(src/common)                                # Common headers
include_directories(${CMAKE_BINARY_DIR})                       # DSL headers
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # CUDA headers

if (MPI_ENABLED)
    find_package(MPI REQUIRED)
    include_directories(${MPI_CXX_INCLUDE_DIRS})
endif()

## Subdirectories
add_subdirectory(src/utils)
add_subdirectory(src/core/kernels)
add_subdirectory(src/core)

if (BUILD_SAMPLES)
    add_subdirectory(samples/standalone)
    add_subdirectory(samples/ctest)
    add_subdirectory(samples/cpptest)
    add_subdirectory(samples/mpitest)
    add_subdirectory(samples/benchmark)
    add_subdirectory(samples/bwtest)
    add_subdirectory(samples/genbenchmarkscripts)
    add_subdirectory(samples/mpi_reduce_bench)
    add_subdirectory(samples/fortrantest)
endif()
