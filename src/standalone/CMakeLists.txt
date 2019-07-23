##############################################
##  CMakeLists.txt for Astaroth Standalone  ##
##############################################

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
add_compile_options(-march=native -pipe ${OpenMP_CXX_FLAGS})
add_compile_options(-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion)# -Wshadow)

## Compile and link
add_executable(ac_run ${SOURCES})
target_link_libraries(ac_run PRIVATE "${OpenMP_CXX_FLAGS}" astaroth_core ${SDL2_LIBRARY})

# Define the config directory
if (ALTER_CONF)
    target_compile_definitions(ac_run PRIVATE CONFIG_PATH="${CMAKE_BINARY_DIR}/")
else()
    target_compile_definitions(ac_run PRIVATE CONFIG_PATH="${CMAKE_SOURCE_DIR}/config/")
endif()
