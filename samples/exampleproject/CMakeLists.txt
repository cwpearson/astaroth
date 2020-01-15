## C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Compilation flags
add_compile_options(-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow)

## Compile and link
add_executable(ac_simulate simulation.cc)
target_link_libraries(ac_simulate PRIVATE astaroth_core astaroth_utils)
add_definitions(-DAC_DEFAULT_CONFIG="${CMAKE_SOURCE_DIR}/config/astaroth.conf")
