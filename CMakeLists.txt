cmake_minimum_required(VERSION 3.16)
project(matvec_omp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(matvec matvec_omp.cpp)
target_compile_options(matvec PRIVATE -O3 -march=native)
target_link_libraries(matvec PRIVATE OpenMP::OpenMP_CXX)