########################################
##  CMakeLists.txt for Astaroth Core  ##
########################################

# Require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3") # Leads to very bad performance on my local machine for some reason

## Find packages
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
# find_package(OpenMP REQUIRED)

## Architecture and optimization flags
set(CUDA_ARCH_FLAGS -gencode arch=compute_37,code=sm_37
                    -gencode arch=compute_50,code=sm_50
                    -gencode arch=compute_60,code=sm_60
                    -gencode arch=compute_61,code=sm_61
                    -gencode arch=compute_70,code=sm_70
                    -lineinfo
                    -ftz=true # Flush denormalized floats to zero
                    -std=c++11
                    )# --compiler-options ${OpenMP_CXX_FLAGS})
                    #--maxrregcount=255
                    # -Xptxas -dlcm=ca opt-in to cache all global loads to L1/texture cache
                    # =cg to opt out

set(CUDA_WARNING_FLAGS --compiler-options -Wall,-Wextra,-Werror,-Wdouble-promotion,-Wfloat-conversion,-Wshadow)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH_FLAGS} ${CUDA_WARNING_FLAGS})

# Workaround for CMake bug https://gitlab.kitware.com/cmake/cmake/issues/16411
if (CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE")
    set(CUDA_NVCC_FLAGS_RELEASE -O3)
elseif (CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
    set(CUDA_NVCC_FLAGS_DEBUG --device-debug --generate-line-info --ptxas-options=-v)
endif()

## Create and link the library
#set(CMAKE_POSITION_INDEPENDENT_CODE ON) # fpic for shared libraries
cuda_add_library(astaroth_core STATIC astaroth.cc node.cc device.cc kernels/boundconds.cu kernels/integration.cu kernels/reductions.cu)
target_include_directories(astaroth_core PRIVATE . ${CUDA_INCLUDE_DIRS})
target_link_libraries(astaroth_core m ${CUDA_LIBRARIES})
target_compile_definitions(astaroth_core PRIVATE AC_USE_CUDA_RUNTIME_API)
add_dependencies(astaroth_core dsl_headers)

## Definitions
if (MULTIGPU_ENABLED)
    add_definitions(-DAC_MULTIGPU_ENABLED=1)
endif ()

if (MPI_ENABLED)
    find_package(MPI REQUIRED)
    target_link_libraries(astaroth_core astaroth_utils MPI::MPI_CXX)
    target_compile_definitions(astaroth_core PRIVATE AC_MPI_ENABLED=1)
endif ()
