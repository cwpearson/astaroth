##############################################
##  CMakeLists.txt for the C API test       ##
##############################################

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(ctest main.c)
target_link_libraries(ctest PRIVATE astaroth_core astaroth_utils)
