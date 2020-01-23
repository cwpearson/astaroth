## Astaroth Core
add_library(astaroth_core STATIC device.cc node.cc astaroth.cc)
target_include_directories(astaroth_core PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(astaroth_core astaroth_utils astaroth_kernels cudart)
