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

# Можно задавать размер матрицы через -DN=40000 в cmake
# По умолчанию (если не передали) будет N=20000 из кода
option(MATVEC_N "Matrix size N for compile-time macro" "")
if (MATVEC_N)
    target_compile_definitions(task1 PRIVATE N=${MATVEC_N})
endif()