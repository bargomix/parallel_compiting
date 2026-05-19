cmake_minimum_required(VERSION 3.16)
project(task3_2_1_thread_pool CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TASK3_2_1_TASKS 100 CACHE STRING "Number of tasks for every client")
set(TASK3_2_1_WORKERS 4 CACHE STRING "Number of server worker threads")
set(TASK3_2_1_RESULTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/results" CACHE PATH "Directory for CSV results")

find_package(Threads REQUIRED)

add_executable(task3_2_1 task3_2_1.cpp)
target_link_libraries(task3_2_1 PRIVATE Threads::Threads)
target_compile_definitions(task3_2_1 PRIVATE
    CLIENT_TASKS=${TASK3_2_1_TASKS}
    SERVER_WORKERS=${TASK3_2_1_WORKERS}
    TASK3_2_1_RESULTS_DIR="${TASK3_2_1_RESULTS_DIR}"
)

add_executable(task3_2_1_test task3_2_1_test.cpp)
target_link_libraries(task3_2_1_test PRIVATE Threads::Threads)
target_compile_definitions(task3_2_1_test PRIVATE
    CLIENT_TASKS=${TASK3_2_1_TASKS}
    TASK3_2_1_RESULTS_DIR="${TASK3_2_1_RESULTS_DIR}"
)
