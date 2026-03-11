cmake_minimum_required(VERSION 3.16)
project(task2_integrate_omp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(task2 task2.cpp)
target_compile_options(task2 PRIVATE -O3 -march=native)
target_link_libraries(task2 PRIVATE OpenMP::OpenMP_CXX)

# Можно менять число шагов через -DINTEG_STEPS=...
option(INTEG_STEPS "Integration steps (compile-time macro NSTEPS)" "")
if (INTEG_STEPS)
    target_compile_definitions(task2 PRIVATE NSTEPS=${INTEG_STEPS})
endif()