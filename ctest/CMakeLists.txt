include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable(ctest main.c)
target_link_libraries(ctest astaroth_core)
