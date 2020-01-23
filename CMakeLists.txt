## CMake settings
cmake_minimum_required(VERSION 3.9) # Required for first-class CUDA support
find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)

## Project settings
project(astaroth C CXX CUDA)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## Project-wide compilation flags
set(COMMON_WARNING_FLAGS "-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow")
set(COMMON_WARNING_FLAGS_CUDA "-Wall,-Wextra,-Werror,-Wdouble-promotion,-Wfloat-conversion,-Wshadow")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_WARNING_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 --compiler-options=${COMMON_WARNING_FLAGS_CUDA}")

## Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release") # Default
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

## Options
option(DOUBLE_PRECISION "Generates double precision code."        OFF)
option(BUILD_SAMPLES    "Builds projects in samples subdirectory" ON)
option(BUILD_STANDALONE "Builds standalone Astaroth"              ON)

## Compile ACC
add_subdirectory(acc)

## Compile AC headers
if (NOT DEFINED DSL_MODULE_DIR)
    set(DSL_MODULE_DIR ${CMAKE_SOURCE_DIR}/acc/mhd_solver) # Default
endif ()
get_filename_component(DSL_MODULE_DIR ${DSL_MODULE_DIR} REALPATH)
message(STATUS "AC module dir: ${DSL_MODULE_DIR}")

set(DSL_SOURCES "${DSL_MODULE_DIR}/*")
set(DSL_HEADERS "${PROJECT_BINARY_DIR}/user_kernels.h"
                "${PROJECT_BINARY_DIR}/user_defines.h")

add_custom_command (
    COMMENT "Building ACC objects ${DSL_MODULE_DIR}"
    COMMAND ${CMAKE_SOURCE_DIR}/acc/compile_acc_module.sh ${DSL_MODULE_DIR}
    DEPENDS ${DSL_SOURCES}
    OUTPUT ${DSL_HEADERS}
)
add_custom_target(dsl_headers ALL DEPENDS ${DSL_HEADERS})
add_dependencies(dsl_headers acc)

## Global flags
if (DOUBLE_PRECISION)
    add_definitions(-DAC_DOUBLE_PRECISION=1)
endif ()

add_definitions(-DAC_DEFAULT_CONFIG="${CMAKE_SOURCE_DIR}/config/astaroth.conf")

## Includes
include_directories(include)             # Library headers
include_directories(src/common)          # Common headers
include_directories(${CMAKE_BINARY_DIR}) # DSL headers

## Subdirectories
add_subdirectory(src/utils)
add_subdirectory(src/core/kernels)
add_subdirectory(src/core)

if (BUILD_SAMPLES)
    add_subdirectory(samples/ctest)
    add_subdirectory(samples/cpptest)
    add_subdirectory(samples/mpitest)
endif()

if (BUILD_STANDALONE)
    add_subdirectory(src/standalone)
endif()
