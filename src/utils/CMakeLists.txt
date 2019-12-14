## C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

## Compilation flags
add_compile_options(-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow)
add_compile_options(-mavx)

## Compile
find_package(OpenMP)
add_library(astaroth_utils STATIC config_loader.c memory.c verification.c modelsolver.c)
target_link_libraries(astaroth_utils PRIVATE astaroth_core m OpenMP::OpenMP_C)
