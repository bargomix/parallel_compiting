#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <cmath>

#define VEC_SIZE 50000000

int main(int argc, char **argv)
{
    double *vec1 = new double[VEC_SIZE];
    double *vec2 = new double[VEC_SIZE];
    double *res  = new double[VEC_SIZE];

    // ✅ Исправление 1: добавлен start_time перед инициализацией
    auto start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < VEC_SIZE; i++) {
        vec1[i] = i + 1;
        vec2[i] = VEC_SIZE - i;
        res[i]  = 0;
    }

    auto end1 = std::chrono::steady_clock::now();

    #pragma omp parallel
    {
        int num_threads     = omp_get_num_threads();
        int size_per_thread = VEC_SIZE / num_threads;
        int thread_id       = omp_get_thread_num();
        int start           = thread_id * size_per_thread;
        int end             = (thread_id == num_threads - 1)
                              ? VEC_SIZE
                              : (thread_id + 1) * size_per_thread;

        for (int i = start; i < end; i++) {
            res[i] += std::pow(std::pow(vec1[i] + vec2[i], 10), 10);
        }
    }

    // ✅ Исправление 2: дописан now() после параллельного блока
    auto end2 = std::chrono::steady_clock::now();

    // ✅ Исправление 3: {} заменены на = для duration
    std::chrono::duration<double> elapsed_seconds1 = end1 - start_time;
    std::chrono::duration<double> elapsed_seconds2 = end2 - end1;

    std::cout << "Init time: " << elapsed_seconds1.count() << " sec" << std::endl;
    std::cout << "Work time: " << elapsed_seconds2.count() << " sec" << std::endl;
    std::cout << "Threads:   " << omp_get_max_threads()             << std::endl;

    delete[] vec1;
    delete[] vec2;
    delete[] res;
    return 0;
}
