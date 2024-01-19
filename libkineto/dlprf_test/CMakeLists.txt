cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

FILE(GLOB SOURCES_CPP DlprofActivityProfilerTest.*)
message(STATUS "source: ${SOURCES_CPP}")

add_library(tmp_lib INTERFACE)
target_include_directories(tmp_lib INTERFACE "${LIBKINETO_DIR}"
        "${LIBKINETO_DIR}/include"
        "${LIBKINETO_DIR}/src"
        "${FMT_SOURCE_DIR}/include"
        "${LIBKINETO_THIRDPARTY_DIR}/googletest/googletest/include"
        "${CUDA_SOURCE_DIR}/include")

target_link_libraries(tmp_lib INTERFACE gtest_main gmock kineto)


set(targets "")
foreach(src ${SOURCES_CPP})
    get_filename_component(exec_name ${src} NAME_WE)
    set(target test-${exec_name})
    add_executable(${target} ${src})
    target_compile_options(${target} PRIVATE "${KINETO_COMPILE_OPTIONS}")
    target_link_libraries(${target} PUBLIC tmp_lib)
endforeach()

add_subdirectory(dlprof)

