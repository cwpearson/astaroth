find_package(CUDAToolkit)

## Astaroth Core
add_library(astaroth_core STATIC device.cc node.cc astaroth.cc)
target_link_libraries(astaroth_core astaroth_utils astaroth_kernels CUDA::cudart)

## Options
if (MPI_ENABLED)
    find_package(MPI)
    find_package(OpenMP)
    target_link_libraries(astaroth_core MPI::MPI_CXX OpenMP::OpenMP_CXX)
endif()

if (MULTIGPU_ENABLED)
    target_compile_definitions(astaroth_core PRIVATE -DAC_MULTIGPU_ENABLED=1)
endif ()
