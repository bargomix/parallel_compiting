cmake_minimum_required(VERSION 3.16)
project(task3_1_threads CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TASK3_1_SIZE 20000 CACHE STRING "Matrix size")
set(TASK3_1_RESULTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/results" CACHE PATH "Directory for CSV results")

find_package(Threads REQUIRED)

add_executable(task3_1 task3_1.cpp)
target_link_libraries(task3_1 PRIVATE Threads::Threads)
target_compile_definitions(task3_1 PRIVATE
    N=${TASK3_1_SIZE}
    TASK3_1_RESULTS_DIR="${TASK3_1_RESULTS_DIR}"
)
