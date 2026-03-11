#include <omp.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>

#ifndef N
#define N 20000
#endif

#ifndef RUNS
#define RUNS 5
#endif

// 0 = min, 1 = median
#ifndef TIME_MODE
#define TIME_MODE 0
#endif

// Детерминированная инициализация — чтобы вычисления не свелись к тривиальному случаю
static inline double initA(int i, int j) {
    return 1.0 + 1e-6 * ((i * 1315423911u) ^ (j * 2654435761u));
}

static inline double initX(int i) {
    return 1.0 + 1e-6 * (i & 0xFFFF);
}

// Последовательная версия: y = A * x
static void matvec_serial(const double* A, const double* x, double* y) {
    for (int i = 0; i < N; ++i) {
        const long long base = 1LL * i * N;
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            sum += A[base + j] * x[j];
        }
        y[i] = sum;
    }
}

// Параллельная версия
static void matvec_omp(const double* A, const double* x, double* y) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const long long base = 1LL * i * N;
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            sum += A[base + j] * x[j];
        }
        y[i] = sum;
    }
}

static double pick_time(double* t, int count) {
    std::sort(t, t + count);

#if TIME_MODE == 0
    return t[0];  // minimum
#else
    if (count % 2 == 1) {
        return t[count / 2]; // median
    } else {
        return 0.5 * (t[count / 2 - 1] + t[count / 2]); // median for even count
    }
#endif
}

// Замер последовательной версии с несколькими прогонами
static double time_serial(const double* A, const double* x, double* y) {
    using clock = std::chrono::steady_clock;

    matvec_serial(A, x, y); // warmup

    double times[RUNS];

    for (int r = 0; r < RUNS; ++r) {
        auto t0 = clock::now();
        matvec_serial(A, x, y);
        auto t1 = clock::now();
        times[r] = std::chrono::duration<double>(t1 - t0).count();
    }

    return pick_time(times, RUNS);
}

// Замер параллельной версии с несколькими прогонами
static double time_parallel(const double* A, const double* x, double* y, int p) {
    using clock = std::chrono::steady_clock;

    omp_set_dynamic(0);
    omp_set_num_threads(p);

    matvec_omp(A, x, y); // warmup

    double times[RUNS];

    for (int r = 0; r < RUNS; ++r) {
        auto t0 = clock::now();
        matvec_omp(A, x, y);
        auto t1 = clock::now();
        times[r] = std::chrono::duration<double>(t1 - t0).count();
    }

    return pick_time(times, RUNS);
}

int main() {
    const int P[] = {1, 2, 4, 6, 8, 16, 20, 40};
    const int Pn = (int)(sizeof(P) / sizeof(P[0]));

    double* A = new double[1LL * N * N];
    double* x = new double[N];
    double* y = new double[N];

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            x[i] = initX(i);
            y[i] = 0.0;
        }

#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            const long long base = 1LL * i * N;
            for (int j = 0; j < N; ++j) {
                A[base + j] = initA(i, j);
            }
        }
    }

    omp_set_dynamic(0);
    omp_set_num_threads(1);
    double T1 = time_serial(A, x, y);

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);

#if TIME_MODE == 0
    const char* mode_name = "min";
#else
    const char* mode_name = "median";
#endif

    std::cout << "N=" << N << " RUNS=" << RUNS << " MODE=" << mode_name << "\n";
    std::cout << "p,Tp,Sp\n";

    std::ofstream f("../results.csv", std::ios::app);
    f << "N=" << N << "\n";
    f << "p,Tp,Sp\n";

    for (int k = 0; k < Pn; ++k) {
        int p = P[k];

        double Tp = (p == 1) ? T1 : time_parallel(A, x, y, p);
        double Sp = T1 / Tp;

        std::cout << p << "," << Tp << "," << Sp << "\n";
        f << p << "," << Tp << "," << Sp << "\n";
    }

    delete[] A;
    delete[] x;
    delete[] y;
    return 0;
}