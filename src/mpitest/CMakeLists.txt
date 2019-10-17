##############################################
##    CMakeLists.txt for the MPI test       ##
##############################################

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mpitest main.cc)
target_link_libraries(mpitest astaroth_core)
