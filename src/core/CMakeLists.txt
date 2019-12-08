########################################
##  CMakeLists.txt for Astaroth Core  ##
########################################

# Require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Require CUDA
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_FLAGS "-gencode arch=compute_70,code=sm_70 -ftz=true --restrict")

# Compile kernels
add_library(astaroth_kernels STATIC kernels/boundconds.cu kernels/integration.cu kernels/reductions.cu)
target_include_directories(astaroth_kernels PRIVATE .)
target_compile_features(astaroth_kernels PRIVATE cxx_std_11)
set_target_properties(astaroth_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(astaroth_kernels dsl_headers)

# Compile core
add_library(astaroth_core STATIC astaroth.cc node.cc device.cc)
target_include_directories(astaroth_core PRIVATE . ${CUDA_INCLUDE_DIRS})
target_link_libraries(astaroth_core m cudart astaroth_kernels)
target_compile_definitions(astaroth_core PRIVATE AC_USE_CUDA_RUNTIME_API)
set_target_properties(astaroth_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(astaroth_kernels dsl_headers)

if (MULTIGPU_ENABLED)
    target_compile_definitions(astaroth_core PRIVATE AC_MULTIGPU_ENABLED=1)
endif ()

if (MPI_ENABLED)
    target_link_libraries(astaroth_core astaroth_utils)
    target_compile_definitions(astaroth_core PRIVATE AC_MPI_ENABLED=1)
endif ()
