###################################
##  CMakeLists.txt for Astaroth  ##
###################################
#
#   Usage: mkdir build && cd build && cmake <options> .. && make
#
#   If you want to see the exact flags used during compilation, compile with
#   "make VERBOSE=1"
#
#   Print all options: cmake -LAH ..
#

## CMake settings
cmake_minimum_required (VERSION 3.5.1) # Need >= 3.8 for first-class CUDA support
find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)

## Project settings
project(astaroth C CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## Options
option(BUILD_DEBUG              "Builds the program with extensive error checking"          OFF)
option(BUILD_STANDALONE         "Builds the standalone Astaroth"                            ON)
option(BUILD_RT_VISUALIZATION   "Builds the module for real-time visualization using SDL2"  OFF)
option(BUILD_C_API_TEST         "Builds a C program to test whether the API is conformant"  OFF)
option(BUILD_MPI_TEST           "Builds a C program to test whether MPI works"              OFF)
option(DOUBLE_PRECISION         "Generates double precision code"                           OFF)
option(MULTIGPU_ENABLED         "If enabled, uses all the available GPUs"                   ON)
option(ALTER_CONF               "If enabled, loads astaroth.conf from the build directory"  OFF)

## Build types
# Available types (case-sensitive):
# RELEASE         (best performance)
# DEBUG           (w/ debug information, non-concurrent kernels)
if (BUILD_DEBUG)
    set(CMAKE_BUILD_TYPE DEBUG)
else ()
    set(CMAKE_BUILD_TYPE RELEASE)
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

## Defines
if (DOUBLE_PRECISION)
    add_definitions(-DAC_DOUBLE_PRECISION=1)
else ()
    add_definitions(-DAC_DOUBLE_PRECISION=0)
endif ()

## Include directories
include_directories(.)
include_directories(include)

## Subdirectories
add_subdirectory(src/core) # The core library

if (BUILD_STANDALONE)
    add_subdirectory(src/standalone)
endif ()

if (BUILD_C_API_TEST)
    add_subdirectory(src/ctest)
endif ()

if (BUILD_MPI_TEST)
    add_subdirectory(src/mpitest)
endif ()
