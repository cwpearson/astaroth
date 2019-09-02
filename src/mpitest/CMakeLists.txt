##############################################
##    CMakeLists.txt for the MPI test       ##
##############################################

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(mpitest main.c)
target_include_directories(mpitest PRIVATE ${CMAKE_SOURCE_DIR}/src/standalone ${MPI_C_INCLUDE_PATH})
target_link_libraries(mpitest astaroth_core astaroth_standalone ${MPI_C_LIBRARIES})
