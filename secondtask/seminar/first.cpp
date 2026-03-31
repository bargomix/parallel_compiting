#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>

// Размеры матрицы (можно менять)
static const int M = 20000;
static const int N = 20000;

// -------------------------------------------------------
// Выделение памяти с проверкой
// -------------------------------------------------------
static double *xmalloc(size_t size)
{
    double *p = (double *)malloc(size);
    if (!p) {
        fprintf(stderr, "Error: malloc failed\n");
        exit(1);
    }
    return p;
}

// -------------------------------------------------------
// Последовательное умножение матрицы на вектор
// c[M] = a[M x N] * b[N]
// -------------------------------------------------------
void matvec_serial(const double *a, const double *b, double *c, int m, int n)
{
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += a[i * n + j] * b[j];
        c[i] = sum;
    }
}

// -------------------------------------------------------
// Параллельное умножение (OpenMP, ручное распределение строк)
// Каждый поток обрабатывает свой диапазон строк [lb, ub]
// -------------------------------------------------------
void matvec_parallel(const double *a, const double *b, double *c, int m, int n)
{
#pragma omp parallel
    {
        int nthreads      = omp_get_num_threads();
        int threadid      = omp_get_thread_num();
        int items         = m / nthreads;          // строк на поток
        int lb            = threadid * items;       // нижняя граница
        int ub            = (threadid == nthreads - 1) ? m - 1
                                                        : lb + items - 1; // верхняя граница

        for (int i = lb; i <= ub; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
                sum += a[i * n + j] * b[j];
            c[i] = sum;
        }
    }
}

// -------------------------------------------------------
// Инициализация данных (параллельная, чтобы first-touch
// распределял страницы памяти по NUMA-узлам правильно)
// -------------------------------------------------------
void init_parallel(double *a, double *b, int m, int n)
{
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items    = m / nthreads;
        int lb       = threadid * items;
        int ub       = (threadid == nthreads - 1) ? m - 1 : lb + items - 1;

        // Инициализация строк матрицы, принадлежащих этому потоку
        for (int i = lb; i <= ub; i++)
            for (int j = 0; j < n; j++)
                a[i * n + j] = (double)(i + j);
    }
    // Вектор b инициализируется главным потоком (он небольшой)
    for (int j = 0; j < n; j++)
        b[j] = (double)j;
}

// -------------------------------------------------------
// Проверка корректности: сравниваем serial и parallel
// -------------------------------------------------------
bool verify(const double *c_serial, const double *c_parallel, int m)
{
    for (int i = 0; i < m; i++) {
        if (fabs(c_serial[i] - c_parallel[i]) > 1e-6) {
            fprintf(stderr, "Mismatch at i=%d: serial=%.6f parallel=%.6f\n",
                    i, c_serial[i], c_parallel[i]);
            return false;
        }
    }
    return true;
}

// -------------------------------------------------------
// main
// -------------------------------------------------------
int main(int argc, char *argv[])
{
    printf("Matrix-vector product: C[%d] = A[%d x %d] * B[%d]\n", M, M, N, N);
    printf("Memory: %.1f GiB\n",
           (double)(M * N + M + N) * sizeof(double) / (1 << 30));

    // Выделяем память
    double *a        = xmalloc((size_t)M * N * sizeof(double));
    double *b        = xmalloc((size_t)N * sizeof(double));
    double *c_serial = xmalloc((size_t)M * sizeof(double));
    double *c_par    = xmalloc((size_t)M * sizeof(double));

    // Параллельная инициализация (NUMA first-touch)
    init_parallel(a, b, M, N);

    // --- Последовательный запуск ---
    double t_serial = omp_get_wtime();
    matvec_serial(a, b, c_serial, M, N);
    t_serial = omp_get_wtime() - t_serial;
    printf("\n[Serial]\n  Time: %.4f sec\n", t_serial);

    // --- Параллельные запуски с разным числом потоков ---
    int thread_counts[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int n_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("\n%-10s %-12s %-10s\n", "Threads", "Time (sec)", "Speedup");
    printf("%-10s %-12s %-10s\n", "-------", "----------", "-------");

    for (int t = 0; t < n_tests; t++) {
        int nthreads = thread_counts[t];
        omp_set_num_threads(nthreads);

        double t_par = omp_get_wtime();
        matvec_parallel(a, b, c_par, M, N);
        t_par = omp_get_wtime() - t_par;

        double speedup = t_serial / t_par;

        // Проверяем только при первом запуске
        if (t == 0) {
            if (verify(c_serial, c_par, M))
                printf("  [OK] Results match\n\n");
            else
                printf("  [FAIL] Results differ!\n\n");
        }

        printf("%-10d %-12.4f %-10.2f\n", nthreads, t_par, speedup);
    }

    free(a);
    free(b);
    free(c_serial);
    free(c_par);
    return 0;
}
