## C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

## Compilation flags
add_compile_options(-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow)

## Compile
add_library(astaroth_utils STATIC config_loader.c memory.c verification.c)
target_link_libraries(astaroth_utils PRIVATE astaroth_core m)
target_compile_definitions(astaroth_utils PUBLIC -DAC_DEFAULT_CONFIG="${CMAKE_SOURCE_DIR}/config/astaroth.conf")
