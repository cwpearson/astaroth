## C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

## Compile
find_package(OpenMP)
add_library(astaroth_utils STATIC config_loader.c memory.c verification.c modelsolver.c)
target_link_libraries(astaroth_utils PRIVATE astaroth_core OpenMP::OpenMP_C)
target_compile_options(astaroth_utils PRIVATE -Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow)
target_compile_options(astaroth_utils PRIVATE -mavx)
