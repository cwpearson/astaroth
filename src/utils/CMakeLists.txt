## Astaroth Utils
add_library(astaroth_utils STATIC config_loader.c memory.c verification.c modelsolver.c modelreduce.c)
add_dependencies(astaroth_utils dsl_headers)
target_compile_options(astaroth_utils PRIVATE "-mavx")
