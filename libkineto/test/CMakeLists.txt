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

