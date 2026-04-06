#include <omp.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>

#ifndef N
#define N 70000
#endif

#ifndef FIXED_THREADS
#define FIXED_THREADS 8
#endif

#ifndef MAX_ITERS
#define MAX_ITERS 60
#endif

#ifndef EPS
#define EPS 1e-5
#endif

struct SchedResult {
    std::string schedule_name;
    int chunk;
    double init_s;
    double work_s;
    double checksum;
    int iters;
    double final_diff;
};

static omp_sched_t parse_schedule(const std::string& name) {
    if (name == "static")  return omp_sched_static;
    if (name == "dynamic") return omp_sched_dynamic;
    if (name == "guided")  return omp_sched_guided;
    return omp_sched_static;
}

static SchedResult run_once(const std::string& sched_name, int chunk) {
    omp_sched_t sched_kind = parse_schedule(sched_name);
    omp_set_schedule(sched_kind, chunk);

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

    return {
        sched_name,
        chunk,
        t_init_end - t_init_start,
        t_work_end - t_work_start,
        checksum,
        iters_done,
        final_diff
    };
}

int main() {
    omp_set_dynamic(0);
    omp_set_num_threads(FIXED_THREADS);

    std::vector<std::string> schedules = {"static", "dynamic", "guided"};
    std::vector<int> chunks = {1, 10, 50, 100, 500, 1000, 5000, 10000};

    std::filesystem::path results_dir = std::filesystem::current_path().parent_path() / "results";
    std::filesystem::create_directories(results_dir);

    std::ofstream f(results_dir / "schedule_results.csv", std::ios::out | std::ios::trunc);
    f << "schedule,chunk,init_time_s,work_time_s,checksum,iters,final_diff\n";

    std::cout << "N = " << N << std::endl;
    std::cout << "threads = " << FIXED_THREADS << std::endl;
    std::cout << std::endl;

    for (const auto& sched : schedules) {
        for (int chunk : chunks) {
            std::cout << "schedule=" << sched << ", chunk=" << chunk << " ..." << std::endl;

            SchedResult r = run_once(sched, chunk);

            std::cout << "  init_time=" << r.init_s << std::endl;
            std::cout << "  work_time=" << r.work_s << std::endl;
            std::cout << "  checksum=" << r.checksum << std::endl;
            std::cout << "  iters=" << r.iters << std::endl;
            std::cout << "  final_diff=" << r.final_diff << std::endl;
            std::cout << std::endl;

            f << r.schedule_name << ","
              << r.chunk << ","
              << r.init_s << ","
              << r.work_s << ","
              << r.checksum << ","
              << r.iters << ","
              << r.final_diff << "\n";
        }
    }

    std::cout << "Saved CSV: " << (results_dir / "schedule_results.csv") << std::endl;
    return 0;
}