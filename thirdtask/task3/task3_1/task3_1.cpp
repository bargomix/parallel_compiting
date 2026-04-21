#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#ifndef N
#define N 20000
#endif

#ifndef TASK3_1_RESULTS_DIR
#define TASK3_1_RESULTS_DIR "."
#endif

struct RunTimes {
    int size;
    int threads;
    double init_s;
    double work_s;
    double checksum;
};

template <class Func>
static void parallel_for(std::size_t total, int threads, Func&& fn) {
    if (total == 0) {
        return;
    }

    int workers = threads;
    if (workers < 1) {
        workers = 1;
    }

    if (static_cast<std::size_t>(workers) > total) {
        workers = static_cast<int>(total);
    }

    std::vector<std::jthread> pool;
    pool.reserve(workers);

    std::size_t base = total / static_cast<std::size_t>(workers);
    std::size_t rem = total % static_cast<std::size_t>(workers);
    std::size_t begin = 0;

    for (int id = 0; id < workers; id++) {
        std::size_t block = base + (static_cast<std::size_t>(id) < rem ? 1 : 0);
        std::size_t end = begin + block;

        pool.emplace_back([begin, end, &fn] {
            fn(begin, end);
        });

        begin = end;
    }
}

static std::vector<int> make_threads_list() {
    return {1, 2, 4, 7, 8, 16, 20, 40};
}

static double make_matrix_value(std::size_t i, std::size_t j) {
    if (i == j) {
        return 2.0;
    }

    return 1.0 / (1.0 + static_cast<double>((i * 17 + j * 31) % 1024));
}

static double make_vector_value(std::size_t i) {
    return 1.0 + static_cast<double>(i % 100) * 0.01;
}

static RunTimes run_once(int n, int threads) {
    std::size_t rows = static_cast<std::size_t>(n);
    std::size_t matrix_size = rows * rows;

    double* a = new double[matrix_size];
    double* x = new double[rows];
    double* y = new double[rows];

    auto start1 = std::chrono::steady_clock::now();

    parallel_for(rows, threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            std::size_t row_offset = i * rows;

            for (std::size_t j = 0; j < rows; j++) {
                a[row_offset + j] = make_matrix_value(i, j);
            }
        }
    });

    parallel_for(rows, threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            x[i] = make_vector_value(i);
            y[i] = 0.0;
        }
    });

    auto end1 = std::chrono::steady_clock::now();
    auto start2 = std::chrono::steady_clock::now();

    parallel_for(rows, threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            std::size_t row_offset = i * rows;
            double sum = 0.0;

            for (std::size_t j = 0; j < rows; j++) {
                sum += a[row_offset + j] * x[j];
            }

            y[i] = sum;
        }
    });

    auto end2 = std::chrono::steady_clock::now();

    double checksum = 0.0;
    for (std::size_t i = 0; i < rows; i++) {
        checksum += y[i];
    }

    delete[] a;
    delete[] x;
    delete[] y;

    const std::chrono::duration<double> elapsed1{end1 - start1};
    const std::chrono::duration<double> elapsed2{end2 - start2};

    return {n, threads, elapsed1.count(), elapsed2.count(), checksum};
}

int main() {
    int n = N;
    std::vector<int> thread_counts = make_threads_list();
    std::vector<RunTimes> all_runs;

    std::filesystem::path results_dir = std::filesystem::path(TASK3_1_RESULTS_DIR);
    std::filesystem::create_directories(results_dir);

    for (int threads : thread_counts) {
        std::cout << "N=" << n << std::endl;
        std::cout << "threads=" << threads << std::endl;

        RunTimes r = run_once(n, threads);
        all_runs.push_back(r);

        std::cout << "init_time=" << r.init_s << std::endl;
        std::cout << "work_time=" << r.work_s << std::endl;
        std::cout << "checksum=" << r.checksum << std::endl;
        std::cout << std::endl;
    }

    std::filesystem::path csv_path = results_dir / "task3_1_scaling.csv";
    bool write_header = !std::filesystem::exists(csv_path) || std::filesystem::file_size(csv_path) == 0;
    std::ofstream f(csv_path, std::ios::out | std::ios::app);

    if (write_header) {
        f << "size,threads,init_time_s,work_time_s,checksum,speedup\n";
    }

    double base_work_time = 0.0;
    for (const RunTimes& run : all_runs) {
        if (run.threads == 1) {
            base_work_time = run.work_s;
            break;
        }
    }

    for (const RunTimes& run : all_runs) {
        double speedup = 0.0;
        if (base_work_time > 0.0) {
            speedup = base_work_time / run.work_s;
        }

        f << run.size << ","
          << run.threads << ","
          << run.init_s << ","
          << run.work_s << ","
          << run.checksum << ","
          << speedup << "\n";
    }

    std::cout << "Saved CSV: " << csv_path << std::endl;
    return 0;
}
