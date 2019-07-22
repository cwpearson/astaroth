########################################
##  CMakeLists.txt for Astaroth Core  ##
########################################

#----------------------Find CUDA-----------------------------------------------#

find_package(CUDA 9 REQUIRED)

#----------------------CUDA settings-------------------------------------------#

set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_PROPAGATE_HOST_FLAGS ON)

#----------------------Setup CUDA compilation flags----------------------------#

# Generate code for the default architecture (Pascal)
set(CUDA_ARCH_FLAGS -gencode arch=compute_37,code=sm_37
                    -gencode arch=compute_50,code=sm_50
                    -gencode arch=compute_60,code=sm_60
                    -gencode arch=compute_61,code=sm_61
                    -lineinfo
                    -ftz=true
                    -std=c++11) #--maxrregcount=255 -ftz=true #ftz = flush denormalized floats to zero
# -Xptxas -dlcm=ca opt-in to cache all global loads to L1/texture cache
# =cg to opt out

# Additional CUDA optimization flags
if (CMAKE_BUILD_TYPE MATCHES RELEASE)
    # Doesn't set any additional flags, see CUDA_NVCC_FLAGS_DEBUG below on how
    # to add more
    set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE})
endif()

# Additional CUDA debug flags
if (CMAKE_BUILD_TYPE MATCHES DEBUG)
    # The debug flags must be set inside this if clause, since either CMake 3.5
    # or nvcc 7.5 is bugged:
    # CMake converts these into empty strings when doing RELEASE build, but nvcc
    # 7.5 fails to parse empty flags.
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};
                               --device-debug;
                               --generate-line-info;
                               --ptxas-options=-v)
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_ARCH_FLAGS}")


message("CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})


#------------------Compile and create a static library-------------------------#
file(GLOB CUDA_SOURCES "*.cu" "kernels/*.cu")

# Use -fPIC if -fpic not supported. Some quick non-scientific tests:
# Without fpic: 4.94 user, 4.04 system, 0:09.88 elapsed
# With fpic: 4.96 user, 4.02 system, 0:09.90 elapsed
# With fPIC: 4.94 user, 4.05 system, 0:10.23 elapsed
CUDA_ADD_LIBRARY(astaroth_core STATIC ${CUDA_SOURCES} OPTIONS --compiler-options "-fpic")
