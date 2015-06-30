cmake_minimum_required(VERSION 3.2)
project(ML)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

add_subdirectory(src)
add_subdirectory(lib)