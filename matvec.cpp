#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

static void* aligned_malloc64(std::size_t bytes) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, 64);
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
    return p;
#endif
}
static void aligned_free64(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

static double now_sec() { return omp_get_wtime(); }

// Простая детерминированная "псевдо-случайность" без rand() (быстрее и повторяемо)
static inline double hash01(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    // [0,1)
    return (x >> 11) * (1.0 / 9007199254740992.0);
}

struct Result {
    int threads{};
    double time_sec{};
    double speedup{};
};

static void matvec(const double* A, const double* x, double* y, int N) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const double* row = A + (std::size_t)i * (std::size_t)N;
        double sum = 0.0;
        // внутр. цикл — последовательный, это нормально: распараллеливаем по строкам
        for (int j = 0; j < N; ++j) sum += row[j] * x[j];
        y[i] = sum;
    }
}

// чтобы компилятор/CPU не выкинул вычисления
static double checksum(const double* y, int N) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) s += y[i] * 1e-12;
    return s;
}

static void usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " N [repeats] [csv_path]\n"
        << "Examples:\n"
        << "  " << argv0 << " 20000 3 results_20000.csv\n"
        << "  " << argv0 << " 40000 2 results_40000.csv\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    const int N = std::stoi(argv[1]);
    const int repeats = (argc >= 3) ? std::stoi(argv[2]) : 3;
    const std::string csv_path = (argc >= 4) ? argv[3] : ("results_" + std::to_string(N) + ".csv");

    if (N <= 0 || repeats <= 0) {
        std::cerr << "Bad args.\n";
        return 2;
    }

    const std::vector<int> thread_list = {1, 2, 4, 7, 8, 16, 20, 40};

    const std::size_t nA = (std::size_t)N * (std::size_t)N;
    const std::size_t bytesA = nA * sizeof(double);
    const std::size_t bytesV = (std::size_t)N * sizeof(double);

    std::cout << "N=" << N << "\n";
    std::cout << "Matrix bytes: " << (bytesA / (1024.0 * 1024.0 * 1024.0)) << " GiB\n";
    std::cout << "Repeats per threads: " << repeats << "\n\n";

    double* A = (double*)aligned_malloc64(bytesA);
    double* x = (double*)aligned_malloc64(bytesV);
    double* y = (double*)aligned_malloc64(bytesV);

    if (!A || !x || !y) {
        std::cerr << "Allocation failed. Need ~" << (bytesA + 2 * bytesV) / (1024.0 * 1024.0 * 1024.0)
                  << " GiB total.\n";
        aligned_free64(A);
        aligned_free64(x);
        aligned_free64(y);
        return 1;
    }

    // Важно для NUMA: "first-touch" — инициализируем параллельно теми потоками, которые потом считают.
    // Чтобы first-touch работал корректно, перед init задаём максимальное число потоков.
    const int max_threads = *std::max_element(thread_list.begin(), thread_list.end());
    omp_set_dynamic(0);
    omp_set_num_threads(max_threads);

#pragma omp parallel
    {
        // x
#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            x[i] = 2.0 * hash01((uint64_t)i + 12345ULL) - 1.0;
            y[i] = 0.0;
        }

        // A
#pragma omp for schedule(static)
        for (std::size_t k = 0; k < nA; ++k) {
            A[k] = 2.0 * hash01(k + 999999ULL) - 1.0;
        }
    }

    // Прогон
    std::vector<Result> results;
    results.reserve(thread_list.size());

    double T1 = -1.0;
    double last_chk = 0.0;

    for (int p : thread_list) {
        omp_set_num_threads(p);

        // небольшой warm-up (переиспользуем матрицу/вектор)
        matvec(A, x, y, N);

        // измерения: берём минимальное время из repeats (обычно самое честное)
        double best = 1e100;
        for (int r = 0; r < repeats; ++r) {
            const double t0 = now_sec();
            matvec(A, x, y, N);
            const double t1 = now_sec();
            best = std::min(best, t1 - t0);
        }

        const double chk = checksum(y, N);
        last_chk = chk;

        if (p == 1) T1 = best;

        Result rr;
        rr.threads = p;
        rr.time_sec = best;
        rr.speedup = (T1 > 0.0) ? (T1 / best) : 1.0;
        results.push_back(rr);

        std::cout << "p=" << std::setw(2) << p
                  << "  T=" << std::fixed << std::setprecision(6) << best << " s"
                  << "  S=" << std::setprecision(3) << rr.speedup
                  << "  chk=" << std::setprecision(6) << chk << "\n";
    }

    std::cout << "\nchecksum(last)=" << std::setprecision(12) << last_chk << "\n";

    // CSV для таблицы/графика
    {
        std::ofstream out(csv_path);
        out << "N,threads,time_sec,speedup\n";
        for (auto& r : results) {
            out << N << "," << r.threads << ","
                << std::setprecision(10) << r.time_sec << ","
                << std::setprecision(10) << r.speedup << "\n";
        }
    }
    std::cout << "\nSaved CSV: " << csv_path << "\n";

    aligned_free64(A);
    aligned_free64(x);
    aligned_free64(y);
    return 0;
}