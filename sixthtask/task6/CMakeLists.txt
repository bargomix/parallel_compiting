cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_COMPILER pgc++)

project(task6_heat LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ACC_TARGET "gpu" CACHE STRING "OpenACC target: gpu, host or multicore")

find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(task6 main.cpp)

target_compile_options(task6 PRIVATE -acc=${ACC_TARGET} -Minfo=all)
target_link_options(task6 PRIVATE -acc=${ACC_TARGET})
target_link_libraries(task6 PRIVATE Boost::program_options)
