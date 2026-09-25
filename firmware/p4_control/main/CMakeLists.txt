idf_component_register(SRCS "main.c"
                       INCLUDE_DIRS "."
                       REQUIRES control_lib esp_driver_uart esp_timer)
target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off)
