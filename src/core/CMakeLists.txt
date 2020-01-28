## Astaroth Core
add_library(astaroth_core STATIC device.cc node.cc astaroth.cc)
target_link_libraries(astaroth_core astaroth_utils astaroth_kernels cudart)

## Options
if (MPI_ENABLED)
    find_package(MPI)
    target_link_libraries(astaroth_core MPI::MPI_CXX)
    target_compile_definitions(astaroth_core PRIVATE -DAC_MPI_ENABLED=1)
endif()

if (MULTIGPU_ENABLED)
    target_compile_definitions(astaroth_core PRIVATE -DAC_MULTIGPU_ENABLED=1)
endif ()
