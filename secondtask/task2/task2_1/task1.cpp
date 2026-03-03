#include <omp.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <math.h>

#ifndef N
#define N 20000
#endif

// Детерминированная инициализация — чтобы вычисления не свелись к тривиальному случаю
// и компилятор не пытался агрессивно упрощать.
static inline double initA(int i, int j) {
    return 1.0 + 1e-6 * ((i * 1315423911u) ^ (j * 2654435761u));
}

static inline double initX(int i) {
    return 1.0 + 1e-6 * (i & 0xFFFF);
}

// Последовательная версия: y = A*x.
// Храним матрицу в одном массиве построчно: A[i*N + j].
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

// Параллельная версия: распараллеливаем по строкам матрицы.
// Это естественно: каждая строка даёт один элемент y[i], пересечений по записи нет.
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

// Замер времени последовательного умножения.
// Первый прогон (warmup) не учитываем: он может быть медленнее из-за прогрева кэша,
// подкачки страниц и “раскрутки” частот.
static double time_serial(const double* A, const double* x, double* y) {
    using clock = std::chrono::steady_clock;

    matvec_serial(A, x, y); // warmup

    auto t0 = clock::now();
    matvec_serial(A, x, y);
    auto t1 = clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// Замер времени параллельного умножения для p потоков.
// Здесь важно фиксировать число потоков, иначе OpenMP может выбрать другое значение.
static double time_parallel(const double* A, const double* x, double* y, int p) {
    using clock = std::chrono::steady_clock;

    omp_set_dynamic(0);      // не даём OpenMP менять число потоков “по ситуации”
    omp_set_num_threads(p);

    matvec_omp(A, x, y); // warmup

    auto t0 = clock::now();
    matvec_omp(A, x, y);
    auto t1 = clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

int main() {
    // Набор потоков, по которым заполняем таблицу и строим график.
    const int P[] = {1, 2, 4, 6, 8, 16, 20, 40};
    const int Pn = (int)(sizeof(P) / sizeof(P[0]));

    // Память: A = N*N, x = N, y = N.
    // Для N=40000 матрица ~ 12.8 GiB, поэтому важно чтобы на сервере было достаточно RAM.
    double* A = new double[1LL * N * N];
    double* x = new double[N];
    double* y = new double[N];

    // Инициализация делается параллельно.
    // На NUMA-системах это даёт first-touch: страницы памяти выделяются ближе к тем потокам,
    // которые их “первые тронули”, и это уменьшает обращения к удалённой памяти.
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

    // T1 меряем один раз и используем как базу для ускорения S(p) = T1 / Tp.
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    double T1 = time_serial(A, x, y);

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);

    std::cout << "N=" << N << " T1=" << T1 << "\n";
    std::cout << "p,Tp,Sp\n";

    // Пишем результаты в файл. Путь ../results.csv удобен, когда бинарник запускается из build/.
    std::ofstream f("../results.csv", std::ios::app);
    f << "N=" << N << "\n";
    f << "p,Tp,Sp\n";

    for (int k = 0; k < Pn; ++k) {
        int p = P[k];

        // Для p=1 можно не запускать OpenMP-версию: Tp = T1.
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