#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <fstream>

#define VEC_SIZE 50000000

int main (int argc, char **argv)
{
    double *vec1 = new double[VEC_SIZE];
    double *vec2 = new double[VEC_SIZE];
    double *res = new double[VEC_SIZE];

    for (int i = 0; i < VEC_SIZE; i++)
    {
        vec1[i] = i + 1;
        vec2[i] = VEC_SIZE - i;
        res[i] = 0;
    }

    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int size_per_thread = VEC_SIZE / num_threads;
        int thread_id = omp_get_thread_num();
        int start = thread_id * size_per_thread;
        int end = thread_id == num_threads - 1 ? VEC_SIZE : (thread_id + 1) * size_per_thread;
        for (int i = start; i < end; i++)
        {
            res[i] += vec1[i] + vec2[i];
        }
    }

    std::ofstream file ("res.txt", std::ios::out);
    for (int i = 0; i < VEC_SIZE; i++)
    {
        file << res[i] << std::endl;
    }

    delete[] vec1;
    delete[] vec2;
    delete[] res;  
}