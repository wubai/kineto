# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# ConfigTest
add_executable(ConfigTest ConfigTest.cpp)
target_compile_options(ConfigTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(ConfigTest PRIVATE gtest_main kineto)
target_include_directories(ConfigTest PRIVATE "${LIBKINETO_DIR}")

# CuptiCallbackApiTest
add_executable(CuptiCallbackApiTest CuptiCallbackApiTest.cpp)
target_compile_options(CuptiCallbackApiTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(CuptiCallbackApiTest PRIVATE gtest_main
                                                    kineto
                                                    "${CUDA_SOURCE_DIR}/lib64/libcudart.so")
target_include_directories(CuptiCallbackApiTest PRIVATE "${LIBKINETO_DIR}"
                                                            "${LIBKINETO_DIR}/include"
                                                            "${CUPTI_INCLUDE_DIR}"
                                                            "${CUDA_SOURCE_DIR}/include")

# CuptiRangeProfilerApiTest
add_executable(CuptiRangeProfilerApiTest CuptiRangeProfilerApiTest.cpp)
target_compile_options(CuptiRangeProfilerApiTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(CuptiRangeProfilerApiTest PRIVATE gtest_main
                                                            kineto
                                                            "${CUDA_SOURCE_DIR}/lib64/libcudart.so")
target_include_directories(CuptiRangeProfilerApiTest PRIVATE "${LIBKINETO_DIR}"
                                                                "${LIBKINETO_DIR}/include"
                                                                "${LIBKINETO_DIR}/src"
                                                                "${CUDA_SOURCE_DIR}/include"
                                                                "${CUPTI_INCLUDE_DIR}")

# CuptiRangeProfilerConfigTest
add_executable(CuptiRangeProfilerConfigTest CuptiRangeProfilerConfigTest.cpp)
target_compile_options(CuptiRangeProfilerConfigTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(CuptiRangeProfilerConfigTest PRIVATE gtest_main
                                                            kineto)
target_include_directories(CuptiRangeProfilerConfigTest PRIVATE "${LIBKINETO_DIR}"
                                                                "${LIBKINETO_DIR}/include")
# CuptiRangeProfilerTest
add_executable(CuptiRangeProfilerTest CuptiRangeProfilerTest.cpp)
target_compile_options(CuptiRangeProfilerTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(CuptiRangeProfilerTest PRIVATE gtest_main
                                                      kineto
                                                      "${CUDA_SOURCE_DIR}/lib64/libcudart.so")
target_include_directories(CuptiRangeProfilerTest PRIVATE "${LIBKINETO_DIR}"
                                                          "${LIBKINETO_DIR}/include"
                                                          "${LIBKINETO_DIR}/src"
                                                          "${CUDA_SOURCE_DIR}/include"
                                                          "${CUPTI_INCLUDE_DIR}")

# CuptiStringsTest
add_executable(CuptiStringsTest CuptiStringsTest.cpp)
target_compile_options(CuptiStringsTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(CuptiStringsTest PRIVATE gtest_main kineto)
target_include_directories(CuptiStringsTest PRIVATE "${LIBKINETO_DIR}"
                                                    "${CUDA_SOURCE_DIR}/include"
                                                    "${CUPTI_INCLUDE_DIR}")

# EventProfilerTest
add_executable(EventProfilerTest EventProfilerTest.cpp)
target_compile_options(EventProfilerTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(EventProfilerTest PRIVATE gtest_main
                                                gmock
                                                kineto
                                                "${CUDA_SOURCE_DIR}/lib64/libcudart.so")
target_include_directories(EventProfilerTest PRIVATE "${LIBKINETO_DIR}"
                                                      "${LIBKINETO_DIR}/include"
                                                      "${CUDA_SOURCE_DIR}/include"
                                                      "${CUPTI_INCLUDE_DIR}")

# LoggerObserverTest
add_executable(LoggerObserverTest LoggerObserverTest.cpp)
target_compile_options(LoggerObserverTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(LoggerObserverTest PRIVATE gtest_main
                                                kineto)
target_include_directories(LoggerObserverTest PRIVATE "${LIBKINETO_DIR}"
                                                      "${LIBKINETO_DIR}/include"
                                                      "${LIBKINETO_DIR}/src")

# PidInfoTest
add_executable(PidInfoTest PidInfoTest.cpp)
target_compile_options(PidInfoTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(PidInfoTest PRIVATE gtest_main
                                                kineto)
target_include_directories(PidInfoTest PRIVATE "${LIBKINETO_DIR}")

# CuptiActivityProfilerTest
add_executable(CuptiActivityProfilerTest CuptiActivityProfilerTest.cpp MockActivitySubProfiler.cpp)
target_compile_options(CuptiActivityProfilerTest PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_link_libraries(CuptiActivityProfilerTest PRIVATE gtest_main
                                                gmock
                                                kineto
                                                folly
                                                "${CUDA_SOURCE_DIR}/lib64/libcudart.so")
target_include_directories(CuptiActivityProfilerTest PRIVATE "${LIBKINETO_DIR}"
                                                              "${LIBKINETO_DIR}/include"
                                                              "${LIBKINETO_DIR}/src"
                                                              "${CUDA_SOURCE_DIR}/include"
                                                              "${CUPTI_INCLUDE_DIR}")