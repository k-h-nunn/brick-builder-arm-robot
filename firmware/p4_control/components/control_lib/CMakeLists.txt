# The same C99 sources the host builds into control_lib.so and control_runner, compiled here as
# an ESP-IDF component. Nothing is copied: the sources are read from firmware/control_lib.
set(BB_LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../control_lib)
file(GLOB BB_SOURCES ${BB_LIB_DIR}/src/*.c)
idf_component_register(SRCS ${BB_SOURCES}
                       INCLUDE_DIRS ${BB_LIB_DIR}/include)
# No fused multiply-add, as on the host, so the rounding is the host's.
target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off)
