cmake_minimum_required(VERSION 3.17) # Required for moder CUDA::cudart linking

find_package(MPI)
find_package(OpenMP)
find_package(CUDAToolkit)

add_executable(bwtest main.c)
add_compile_options(-O3)
target_link_libraries(bwtest MPI::MPI_C OpenMP::OpenMP_C CUDA::cudart)
