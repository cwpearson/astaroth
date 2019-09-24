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

## Compile the Astaroth Code compiler
find_package(FLEX)
find_package(BISON)
execute_process (
    COMMAND ./build_acc.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/acc
)
# NOTE: Manually defined DSL_MODULE_DIR must be set relative to the project root, not the actual
# build directory!
# NO! ../acc/mhd_solver
# YES! acc/mhd_solver

if (NOT DEFINED DSL_MODULE_DIR)
    set(DSL_MODULE_DIR ${CMAKE_SOURCE_DIR}/acc/mhd_solver)
endif ()
get_filename_component(DSL_MODULE_DIR ${DSL_MODULE_DIR} REALPATH)
message(STATUS "DSL module dir: ${DSL_MODULE_DIR}")
set(DSL_SOURCES ${DSL_MODULE_DIR}/*)
set(DSL_HEADERS "${PROJECT_BINARY_DIR}/stencil_assembly.cuh"
                "${PROJECT_BINARY_DIR}/stencil_process.cuh"
                "${PROJECT_BINARY_DIR}/stencil_defines.h")
add_custom_command (
    COMMENT "Building AC objects ${DSL_MODULE_DIR}"
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/compile_acc_module.sh ${DSL_MODULE_DIR}
    DEPENDS ${DSL_SOURCES}
    OUTPUT ${DSL_HEADERS}
)
add_custom_target(dsl_headers ALL DEPENDS ${DSL_HEADERS})

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

if (BUILD_RT_VISUALIZATION)
## SDL2
execute_process (
    COMMAND ./setup_dependencies.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty
)
add_subdirectory(3rdparty/SDL2)
endif ()

## Defines
if (DOUBLE_PRECISION)
    add_definitions(-DAC_DOUBLE_PRECISION=1)
else ()
    add_definitions(-DAC_DOUBLE_PRECISION=0)
endif ()

## Include directories
include_directories(.)
include_directories(include)
include_directories(${CMAKE_BINARY_DIR})

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
