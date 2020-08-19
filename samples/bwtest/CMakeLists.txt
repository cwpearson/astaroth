cmake_minimum_required(VERSION 3.17) # Required for moder CUDA::cudart linking

find_package(MPI COMPONENTS C)
find_package(OpenMP)
find_package(CUDAToolkit)

add_executable(bwtest main.c)
target_link_libraries(bwtest MPI::MPI_C OpenMP::OpenMP_C CUDA::cudart_static CUDA::cuda_driver)
target_compile_options(bwtest PRIVATE -O3)
