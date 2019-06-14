################################
##  CMakeLists.txt for utils  ##
################################

file (GLOB SOURCES "*.cc" "model/*.cc")

add_library(astaroth_standalone STATIC ${SOURCES})
target_include_directories(astaroth_standalone PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_compile_definitions(astaroth_standalone PRIVATE CONFIG_PATH=\"${CMAKE_SOURCE_DIR}/config/\")
target_compile_definitions(astaroth_standalone PRIVATE CONFIG_PATH=\"${ASTAROTH_CONF_PATH}\")
