# p4_control: the run-time controller (`firmware/control_lib`) on the ESP32-P4, speaking the
# wire protocol over UART0. Build only from this directory:
#
#   . <esp-idf>/export.sh && idf.py set-target esp32p4 && idf.py build
#
cmake_minimum_required(VERSION 3.16)
set(EXTRA_COMPONENT_DIRS ${CMAKE_CURRENT_LIST_DIR}/components)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(p4_control)
