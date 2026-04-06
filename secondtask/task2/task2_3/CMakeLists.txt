cmake_minimum_required(VERSION 3.16)
project(lab2_openmp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(lab2_var1 lab2_var1.cpp)
add_executable(lab2_var2 lab2_var2.cpp)
add_executable(schedule schedule.cpp)

target_link_libraries(lab2_var1 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(lab2_var2 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(lab2_schedule PRIVATE OpenMP::OpenMP_CXX)