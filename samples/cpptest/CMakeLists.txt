## C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Compilation flags
add_compile_options(-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow)

## Compile and link
add_executable(cpptest main.cc)
target_link_libraries(cpptest PRIVATE astaroth_core astaroth_utils)
