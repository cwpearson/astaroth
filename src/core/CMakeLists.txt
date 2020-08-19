find_package(CUDAToolkit)

## Astaroth Core
add_library(astaroth_core STATIC device.cc node.cc astaroth.cc astaroth_fortran.cc)
target_link_libraries(astaroth_core astaroth_kernels CUDA::cudart CUDA::cuda_driver)

## Options
if (MPI_ENABLED)
    target_link_libraries(astaroth_core MPI::MPI_CXX)
endif()
