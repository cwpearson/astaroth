##############################################
##  CMakeLists.txt for Astaroth Standalone  ##
##############################################

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Files
file (GLOB SOURCES "*.cc" "model/*.cc")

## Find packages
find_package(OpenMP REQUIRED)
if (BUILD_RT_VISUALIZATION)
    add_definitions(-DAC_BUILD_RT_VISUALIZATION=1)
    # SDL 2
    set(SDL2_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/SDL2/include/)
    set(SDL2_LIBRARY_DIR ${CMAKE_SOURCE_DIR}/3rdparty/SDL2/build/)
    set(SDL2_LIBRARY "SDL2")
    include_directories(${SDL2_INCLUDE_DIR})
    link_directories(${SDL2_LIBRARY_DIR})
endif ()

## Compilation flags
add_compile_options(-pipe ${OpenMP_CXX_FLAGS})
add_compile_options(-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion)# -Wshadow)

## Compile and link
add_library(astaroth_standalone STATIC ${SOURCES})
target_link_libraries(astaroth_standalone PRIVATE astaroth_core "${OpenMP_CXX_FLAGS}" ${SDL2_LIBRARY})

add_executable(ac_run main.cc)
target_link_libraries(ac_run PRIVATE astaroth_standalone)
add_definitions(-DAC_DEFAULT_CONFIG="${CMAKE_SOURCE_DIR}/config/astaroth.conf")
