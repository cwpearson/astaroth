enable_language(Fortran)

add_executable(fortrantest main.f90)
target_link_libraries(fortrantest astaroth_core)
