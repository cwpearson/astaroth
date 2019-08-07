########################################
##  CMakeLists.txt for Astaroth Core  ##
########################################

## Find packages
find_package(CUDA 9 REQUIRED)

## Architecture and optimization flags
set(CUDA_ARCH_FLAGS -gencode arch=compute_37,code=sm_37
                    -gencode arch=compute_50,code=sm_50
                    -gencode arch=compute_60,code=sm_60
                    -gencode arch=compute_61,code=sm_61
                    -lineinfo
                    -ftz=true # Flush denormalized floats to zero
                    -std=c++11)
                    #--maxrregcount=255
                    # -Xptxas -dlcm=ca opt-in to cache all global loads to L1/texture cache
                    # =cg to opt out


set(CUDA_WARNING_FLAGS --compiler-options -Wall,-Wextra,-Werror,-Wdouble-promotion,-Wfloat-conversion) # -Wshadow

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH_FLAGS} ${CUDA_WARNING_FLAGS})
set(CUDA_NVCC_FLAGS_RELEASE)
set(CUDA_NVCC_FLAGS_DEBUG --device-debug --generate-line-info --ptxas-options=-v)

## Definitions
if (MULTIGPU_ENABLED)
    add_definitions(-DAC_MULTIGPU_ENABLED=1)
else ()
    add_definitions(-DAC_MULTIGPU_ENABLED=0)
endif ()

## Create and link the library
include_directories(.)
cuda_add_library(astaroth_core STATIC astaroth.cu device.cu)
target_link_libraries(astaroth_core m)
