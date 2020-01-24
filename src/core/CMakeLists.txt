## Astaroth Core
add_library(astaroth_core STATIC device.cc node.cc astaroth.cc)
target_link_libraries(astaroth_core astaroth_utils astaroth_kernels cudart)
