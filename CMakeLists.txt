#
# CMakeLists.txt for generating the makefile for Astaroth.
#   Usage: mkdir build && cd build && cmake <optional flags> ..
#
#   For example: cmake -DDOUBLE_PRECISION=ON ..
#
#   If you want to see the exact flags used during compilation, run
#   "make -j VERBOSE=1"
#
# Make sure your machine satisfies the system requirements:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements

#-------------------General---------------------------------------------------#

project(ASTAROTH_2.0 CXX)
set (CMAKE_CXX_STANDARD 98)
cmake_minimum_required (VERSION 3.5.1) # Need >= 3.8 for first-class CUDA support
cmake_policy (SET CMP0023 NEW)


#-------------------Set user options with default values---------------------#

#Usage f.ex. cmake -DBUILD_DEBUG=ON ..
option(BUILD_DEBUG "Builds the program with extensive error checking" OFF)
option(BUILD_STANDALONE "Builds standalone Astaroth" ON)
option(DOUBLE_PRECISION "Generates double precision code" OFF)
option(TIARA_CLUSTER "Special settings for compilation TIARA GPU cluster" OFF)
option(MULTIGPU_ENABLED "If enabled, uses all the available GPUs" ON)
option(ALTER_CONF "If enabled, loads astaroth.conf from the build directory" OFF)

#-------------------Determine build type--------------------------------------#

#Available types (case-sensitive):
#RELEASE         (best performance)
#DEBUG           (w/ debug information, non-concurrent kernels)
if (BUILD_DEBUG)
    set(CMAKE_BUILD_TYPE DEBUG)
else ()
    set(CMAKE_BUILD_TYPE RELEASE)
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})


#----------------------Find packages------------------------------------------#

# C++ compiler info
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER_ID})

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4)
        message(FATAL_ERROR "GCC version 5.4 or higher required")    
    endif()
endif()

# SDL 2
set(SDL2_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/SDL2/include/)
set(SDL2_LIBRARY_DIR ${CMAKE_SOURCE_DIR}/3rdparty/SDL2/build/)
set(SDL2_LIBRARY "SDL2")
include_directories(${SDL2_INCLUDE_DIR})
link_directories(${SDL2_LIBRARY_DIR})

# CUDA
find_package(CUDA)
if (NOT CUDA_FOUND)
    # find_package(CUDA REQUIRED) gives a confusing error message if it fails,
    # therefore we print the reason here explicitly
    message(FATAL_ERROR "CUDA not found")
endif()
include_directories(${CUDA_INCLUDE_DIRS})

# OpenMP
find_package(OpenMP)
if (NOT OPENMP_FOUND)
    message(WARNING "OpenMP not found. All host-side concurrency disabled \
                    (lower performance).")
else ()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#----------------------Compilation settings-----------------------------------#

#Debug and verification
#set(CMAKE_VERBOSE_MAKEFILE OFF)
#set(CXX_VERBOSE_BUILD OFF)
#set(CUDA_VERBOSE_BUILD OFF)
#include(CTest)
#add_test(ac_test ac_run)
#find_program(MEMORYCHECK_COMMAND valgrind)
#set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full" )


#----------------------Setup defines------------------------------------------#

if (DOUBLE_PRECISION)
	add_definitions(-DAC_DOUBLE_PRECISION=1)
else()
    add_definitions(-DAC_DOUBLE_PRECISION=0)
endif()

# A full integration step is benchmarked by default, use this flag to override and
# benchmark RK3 only
if (GEN_BENCHMARK_RK3)
    add_definitions(-DGEN_BENCHMARK_RK3=1)
else()
    add_definitions(-DGEN_BENCHMARK_RK3=0)
endif()

if (MULTIGPU_ENABLED)
    add_definitions(-DAC_MULTIGPU_ENABLED=1)
else()
    add_definitions(-DAC_MULTIGPU_ENABLED=0)
endif()

#-----------------------TIARA specific options--------------------------------#
#OLD#set (CXX_FLAGS_TIARA "-I/software/opt/cuda/9.0/include/")
# %JP: NOTE! This should not be needed anymore because the command
#      find_package(CUDA) above should find and include this directory automatically
#USE THIS:
if (TIARA_CLUSTER)
	set (CXX_FLAGS_TIARA "-mno-bmi2")
endif()

#----------------------Setup CXX compilation flags----------------------------#
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}\
                             -O2 -march=native -pipe")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}\
                             -O0 -g")

set (CXX_FLAGS_WARNING "-Wall -Wextra -Werror -Wno-error=unused-parameter\
                        -Wno-error=unused-function -Wno-error=unknown-pragmas")

# Also warn about implicit conversions if the compiler supports it
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set (CXX_FLAGS_WARNING "${CXX_FLAGS_WARNING} -Wdouble-promotion -Wfloat-conversion")
endif()

# Other flags. -D_FORCE_INLINES is a workaround to some CUDA/C++ "feature"
# which botches the compilation ("memcpy was not declared in this scope")
# (Not required with cc >= 3.0)
#set(CXX_FLAGS_ETC "-D_FORCE_INLINES")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}\
                    ${CXX_FLAGS_WARNING}\
                    ${CXX_FLAGS_ETC}\
                    ${CXX_FLAGS_TIARA}") # %JP: CXX_FLAGS_TIARA should not be needed,
					 #      see comments in "TIARA specific options"

message("CXX_FLAGS: " ${CMAKE_CXX_FLAGS})


#----------------------Setup core subdirectories------------------------------#

#Include root directory (.) so that the following modules can include their
#parent dir (f.ex. #include "common/stuff.h" instead of "../common/stuff")
include_directories(.)
include_directories(include)
include_directories(src)

# CUDA sources
add_subdirectory(src/core)

#----------------------Link---------------------------------------------------#

if (BUILD_STANDALONE)
    #Define the config directory
    if (ALTER_CONF)
        set(ASTAROTH_CONF_PATH "${CMAKE_BINARY_DIR}/")
    else()
        set(ASTAROTH_CONF_PATH "${CMAKE_SOURCE_DIR}/config/")
    endif()

    #Add additional subdirectories
    add_subdirectory (src/standalone)
    cuda_add_executable(ac_run src/standalone/main.cc)
    target_link_libraries(ac_run astaroth_standalone astaroth_core ${SDL2_LIBRARY})
endif()
