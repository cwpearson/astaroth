## Astaroth Kernels
add_library(astaroth_kernels STATIC kernels.cu)
add_dependencies(astaroth_kernels dsl_headers)
target_compile_features(astaroth_kernels PRIVATE cxx_std_11)
