#include <omp.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <filesystem>

#ifndef N
#define N 70000
#endif

#ifndef MAX_ITERS
#define MAX_ITERS 60
#endif

#ifndef EPS
#define EPS 1e-5
#endif

#ifndef CHUNK
#define CHUNK 0
#endif

#ifndef SCHED_KIND
#define SCHED_KIND omp_sched_static
#endif

struct RunTimes {
    double init_s;
    double work_s;
    double checksum;
    double final_diff;
    int iters;
};

static std::vector<int> make_threads_list() {
    std::vector<int> base = {1, 2, 4, 7, 8, 16, 20, 40};
    int max_threads = omp_get_num_procs();
    std::vector<int> out;

    for (int t : base) {
        if (t <= max_threads) {
            out.push_back(t);
        }
    }

    return out;
}

static RunTimes run_once() {
    double* b = new double[N];
    double* x = new double[N];
    double* x_new = new double[N];

    double tau = 1.0 / (2.0 * (double)(N + 1));

    double t_init_start = 0.0;
    double t_init_end = 0.0;
    double t_work_start = 0.0;
    double t_work_end = 0.0;

    double final_diff = 0.0;
    int iters_done = 0;
    double checksum = 0.0;
    double diff = 0.0;
    int stop = 0;

    #pragma omp parallel shared(b, x, x_new, final_diff, iters_done, checksum, t_init_start, t_init_end, t_work_start, t_work_end, diff, stop)
    {
        #pragma omp single
        {
            std::cout << "threads=" << omp_get_num_threads() << std::endl;
            t_init_start = omp_get_wtime();
        }

        #pragma omp for schedule(runtime)
        for (int i = 0; i < N; i++) {
            b[i] = (double)(N + 1);
        }

        #pragma omp for schedule(runtime)
        for (int i = 0; i < N; i++) {
            x[i] = 0.0;
        }

        #pragma omp for schedule(runtime)
        for (int i = 0; i < N; i++) {
            x_new[i] = 0.0;
        }

        #pragma omp single
        {
            t_init_end = omp_get_wtime();
            t_work_start = t_init_end;
        }

        for (int it = 0; it < MAX_ITERS; it++) {
            #pragma omp single
            {
                diff = 0.0;
                stop = 0;
            }

            #pragma omp barrier

            #pragma omp for reduction(max:diff) schedule(runtime)
            for (int i = 0; i < N; i++) {
                double ax = 0.0;

                for (int j = 0; j < N; j++) {
                    if (j == i) ax += 2.0 * x[j];
                    else        ax += x[j];
                }

                double val = x[i] - tau * (ax - b[i]);
                double cur = std::fabs(val - x[i]);

                x_new[i] = val;
                if (cur > diff) diff = cur;
            }

            #pragma omp for schedule(runtime)
            for (int i = 0; i < N; i++) {
                x[i] = x_new[i];
            }

            #pragma omp single
            {
                final_diff = diff;
                iters_done = it + 1;
                if (diff < EPS) stop = 1;
            }

            #pragma omp barrier
            if (stop) break;
            #pragma omp barrier
        }

        #pragma omp for reduction(+:checksum) schedule(runtime)
        for (int i = 0; i < N; i++) {
            checksum += x[i];
        }

        #pragma omp single
        {
            t_work_end = omp_get_wtime();
        }
    }

    delete[] b;
    delete[] x;
    delete[] x_new;

    return {t_init_end - t_init_start, t_work_end - t_work_start, checksum, final_diff, iters_done};
}

int main() {
    omp_set_dynamic(0);
    omp_set_schedule(SCHED_KIND, CHUNK);

    std::vector<int> Ts = make_threads_list();

    std::filesystem::create_directories("results");
    std::ofstream f("results/lab2_var2_threads.csv", std::ios::out | std::ios::trunc);

    f << "threads,init_time_s,work_time_s,checksum,iters,final_diff\n";

    for (int t : Ts) {
        omp_set_num_threads(t);
        RunTimes r = run_once();

        std::cout << "N=" << N << std::endl;
        std::cout << "init_time=" << r.init_s << std::endl;
        std::cout << "work_time=" << r.work_s << std::endl;
        std::cout << "checksum=" << r.checksum << std::endl;
        std::cout << "iters=" << r.iters << std::endl;
        std::cout << "final_diff=" << r.final_diff << std::endl;
        std::cout << std::endl;

        f << t << ","
          << r.init_s << ","
          << r.work_s << ","
          << r.checksum << ","
          << r.iters << ","
          << r.final_diff << "\n";
    }

    std::cout << "Saved CSV: results/lab2_var2_threads.csv" << std::endl;
    return 0;
}
