cmake_minimum_required(VERSION 3.16)
project(task1_matvec_omp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(task1 task1.cpp)

target_compile_options(task1 PRIVATE
    -O3
    -march=native
)

target_link_libraries(task1 PRIVATE OpenMP::OpenMP_CXX)

option(MATVEC_N "Matrix size N for compile-time macro" "")
if(MATVEC_N)
    target_compile_definitions(task1 PRIVATE N=${MATVEC_N})
endif()

option(MATVEC_RUNS "Number of benchmark runs" "")
if(MATVEC_RUNS)
    target_compile_definitions(task1 PRIVATE RUNS=${MATVEC_RUNS})
endif()

option(MATVEC_TIME_MODE "0=min, 1=median" "")
if(MATVEC_TIME_MODE)
    target_compile_definitions(task1 PRIVATE TIME_MODE=${MATVEC_TIME_MODE})
endif()