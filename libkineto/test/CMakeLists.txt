cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(SUB_PROFILER "${CMAKE_CURRENT_SOURCE_DIR}/MockActivitySubProfiler.cpp")


FILE(GLOB SOURCES_CPP *.cpp)
message(STATUS "source: ${SOURCES_CPP}")
message(STATUS "SUB_PROFILER: ${SUB_PROFILER}")

list(FIND SOURCES_CPP ${SUB_PROFILER} index)

message(STATUS "index: ${index}")

list(REMOVE_AT SOURCES_CPP ${index})

message(STATUS "SOURCES_CPP later: ${SOURCES_CPP}")

add_library(intermediate_lib INTERFACE)
target_include_directories(intermediate_lib INTERFACE "${LIBKINETO_DIR}"
        "${LIBKINETO_DIR}/include"
        "${LIBKINETO_DIR}/src"
        "${FMT_SOURCE_DIR}/include"
        "${LIBKINETO_THIRDPARTY_DIR}/googletest/googletest/include"
        "${CUPTI_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CUDA_SOURCE_DIR}/include")

target_link_libraries(intermediate_lib INTERFACE gtest_main gmock kineto "${CUDA_SOURCE_DIR}/lib64/libcudart.so")

add_library(subprofiler STATIC MockActivitySubProfiler.cpp)
target_include_directories(subprofiler PUBLIC "${CMAKE_SOURCE_DIR}")
target_include_directories(subprofiler PUBLIC "${LIBKINETO_THIRDPARTY_DIR}/fmt/include")
target_include_directories(subprofiler PUBLIC "${LIBKINETO_SOURCE_DIR}")
target_include_directories(subprofiler PUBLIC "${LIBKINETO_INCLUDE_DIR}")
target_include_directories(subprofiler PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_compile_options(subprofiler PRIVATE "${KINETO_COMPILE_OPTIONS}")

set(targets "")
foreach(src ${SOURCES_CPP})
    get_filename_component(exec_name ${src} NAME_WE)
    set(target test-${exec_name})
    add_executable(${target} ${src})
    target_compile_options(${target} PRIVATE "${KINETO_COMPILE_OPTIONS}")
    target_link_libraries(${target} PUBLIC intermediate_lib subprofiler)
endforeach()
