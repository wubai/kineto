cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

FILE(GLOB SOURCES_CPP *.cpp)
message(STATUS "source: ${SOURCES_CPP}")

set(SDK "/LocalRun/dl/v2/sdk")


add_executable(test_dl DlProfTest.cpp)
target_include_directories(test_dl PUBLIC "${SDK}/include")
target_link_libraries(test_dl PUBLIC gtest_main
                                     "${SDK}/lib/libdlhc2.so"
                                     "${SDK}/lib/libcurt.so"
                                     "${SDK}/lib/libhcrt.so")
