#include <omp.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <math.h>

#ifndef NSTEPS
#define NSTEPS 40000000LL
#endif

static inline double f(double x) {
    return 4.0 / (1.0 + x * x);
}

// Последовательная версия
static double integrate_serial(long long nsteps) {
    double step = 1.0 / (double)nsteps;
    double sum = 0.0;
    for (long long i = 0; i < nsteps; ++i) {
        double x = (i + 0.5) * step;
        sum += f(x);
    }
    return sum * step;
}

// OpenMP версия: локальная переменная + atomic (как требуют)
static double integrate_omp(long long nsteps) {
    double step = 1.0 / (double)nsteps;
    double sum = 0.0;

#pragma omp parallel
    {
        double local = 0.0;

#pragma omp for schedule(static)
        for (long long i = 0; i < nsteps; ++i) {
            double x = (i + 0.5) * step;
            local += f(x);
        }

#pragma omp atomic
        sum += local;
    }

    return sum * step;
}

// среднее с отбрасыванием максимума (по методичке измерений)
static double avg_drop_max(double *t, int n) {
    double sum = 0.0;
    double mx = t[0];
    for (int i = 0; i < n; ++i) {
        sum += t[i];
        if (t[i] > mx) mx = t[i];
    }
    return (sum - mx) / (n - 1);
}

static double time_serial(long long nsteps) {
    using clock = std::chrono::steady_clock;
    const int WARMUP = 1;
    const int REPEATS = 6; // усредняем по 5 после отброса max
    double t[REPEATS];

    for (int w = 0; w < WARMUP; ++w) (void)integrate_serial(nsteps);

    for (int k = 0; k < REPEATS; ++k) {
        auto a = clock::now();
        (void)integrate_serial(nsteps);
        auto b = clock::now();
        t[k] = std::chrono::duration<double>(b - a).count();
    }
    return avg_drop_max(t, REPEATS);
}

static double time_parallel(long long nsteps, int p) {
    using clock = std::chrono::steady_clock;
    const int WARMUP = 1;
    const int REPEATS = 6;
    double t[REPEATS];

    omp_set_dynamic(0);
    omp_set_num_threads(p);

    for (int w = 0; w < WARMUP; ++w) (void)integrate_omp(nsteps);

    for (int k = 0; k < REPEATS; ++k) {
        auto a = clock::now();
        (void)integrate_omp(nsteps);
        auto b = clock::now();
        t[k] = std::chrono::duration<double>(b - a).count();
    }
    return avg_drop_max(t, REPEATS);
}

int main() {
    const long long nsteps = NSTEPS;
    const int P[] = {1, 2, 4, 7, 8, 16, 20, 40};
    const int Pn = (int)(sizeof(P) / sizeof(P[0]));

    // меряем T1 один раз
    double T1 = time_serial(nsteps);

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);

    std::cout << "NSTEPS=" << nsteps << " T1=" << T1 << "\n";
    std::cout << "p,Tp,Sp,pi\n";

    std::ofstream f("../results.csv");
    f.setf(std::ios::fixed);
    f.precision(6);
    f << "NSTEPS=" << nsteps << "\n";
    f << "T1=" << T1 << "\n";
    f << "p,Tp,Sp,pi\n";

    for (int k = 0; k < Pn; ++k) {
        int p = P[k];
        double Tp = (p == 1) ? T1 : time_parallel(nsteps, p);
        double Sp = T1 / Tp;

        // значение интеграла (π) для контроля
        omp_set_dynamic(0);
        omp_set_num_threads(p);
        double pi = (p == 1) ? integrate_serial(nsteps) : integrate_omp(nsteps);

        std::cout << p << "," << Tp << "," << Sp << "," << pi << "\n";
        f << p << "," << Tp << "," << Sp << "," << pi << "\n";
    }

    return 0;
}