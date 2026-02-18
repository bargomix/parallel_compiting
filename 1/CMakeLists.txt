cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)

# Опция для выбора типа массива
option(USE_FLOAT "Use float instead of double" OFF)

add_executable(sin_array 1.cpp)

if(USE_FLOAT)
    target_compile_definitions(sin_array PRIVATE ARRAY_TYPE=float USE_FLOAT)
    message(STATUS "Building with float")
else()
    target_compile_definitions(sin_array PRIVATE ARRAY_TYPE=double)
    message(STATUS "Building with double")
endif()