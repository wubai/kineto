# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")

target_link_libraries(intermediate_lib INTERFACE gtest_main gmock kineto
        "${CUDA_SOURCE_DIR}/lib64/libcudart.so"
        "${CUDA_cupti_LIBRARY}"
        "folly")

#target_link_libraries(intermediate_lib INTERFACE gtest_main gmock kineto "${CUDA_SOURCE_DIR}/lib64/libcudart.so" "${CUDA_cupti_LIBRARY}"
#        "/LocalRun/jian.yuan/tmp/myfolly/folly/lib/libfolly.a"
#        "/tmp/fbcode_builder_getdeps-ZLocalRunZjian.yuanZtmpZfollyZbuildZfbcode_builder/installed/double-conversion-ap7rFqTcNKLu3ZwL2PW_dwd_5q0qmlI556EUcJo4yCE/lib/libdouble-conversion.a"
#        "/tmp/fbcode_builder_getdeps-ZLocalRunZjian.yuanZtmpZfollyZbuildZfbcode_builder/installed/fmt-xw5hureOP-rQOeiIYeogmxc945iK18o9yCSqLRND3EY/lib/libfmt.a"
#        "/tmp/fbcode_builder_getdeps-ZLocalRunZjian.yuanZtmpZfollyZbuildZfbcode_builder/installed/googletest-LGXkL97p932UfdsEAvfD9Z2nSdfvWU0QZAffia04-kw/lib/libgtest_main.a"
#        "/tmp/fbcode_builder_getdeps-ZLocalRunZjian.yuanZtmpZfollyZbuildZfbcode_builder/installed/glog-PRKSYh6TUcjSF5WU1TyIpWgmm4UXKLguQ51zbzPh4L0/lib/libglog.so")
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

enable_language(CUDA)
add_executable(test-CuptiProfilerApiTest CuptiProfilerApiTest.cu)
target_compile_definitions(test-CuptiProfilerApiTest PRIVATE HAS_CUPTI_RANGE_PROFILER=1)
target_compile_options(test-CuptiProfilerApiTest  PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(test-CuptiProfilerApiTest PUBLIC intermediate_lib cuda ${CUDA_nvperf_host_LIBRARY}
                       ${CUDA_nvperf_target_LIBRARY})