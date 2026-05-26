#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace po = boost::program_options;

static int idx(int i, int j, int n) {
    return i * n + j;
}

static double linear_interp(double a, double b, double t) {
    return a + (b - a) * t;
}

static void init_grid(std::vector<double>& a, std::vector<double>& anew, int n) {
    const double c00 = 10.0;
    const double c01 = 20.0;
    const double c11 = 30.0;
    const double c10 = 20.0;

    std::fill(a.begin(), a.end(), 0.0);
    std::fill(anew.begin(), anew.end(), 0.0);

    for (int i = 0; i < n; i++) {
        double t = static_cast<double>(i) / (n - 1);
        a[idx(i, 0, n)] = anew[idx(i, 0, n)] = linear_interp(c00, c10, t);
        a[idx(i, n - 1, n)] = anew[idx(i, n - 1, n)] = linear_interp(c01, c11, t);
    }

    for (int j = 0; j < n; j++) {
        double t = static_cast<double>(j) / (n - 1);
        a[idx(0, j, n)] = anew[idx(0, j, n)] = linear_interp(c00, c01, t);
        a[idx(n - 1, j, n)] = anew[idx(n - 1, j, n)] = linear_interp(c10, c11, t);
    }
}

static void print_grid(const std::vector<double>& a, int n) {
    if (n != 10 && n != 13) {
        return;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << a[idx(i, j, n)] << " ";
        }
        std::cout << "\n";
    }
}

static po::variables_map parse_args(int argc, char** argv) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "show help")
        ("size,n", po::value<int>()->default_value(128), "grid size")
        ("eps,e", po::value<double>()->default_value(1.0e-6), "target error")
        ("iters,i", po::value<int>()->default_value(1000000), "max iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        std::exit(0);
    }

    return vm;
}

int main(int argc, char** argv) {
    po::variables_map args = parse_args(argc, argv);

    const int n = args["size"].as<int>();
    const double eps = args["eps"].as<double>();
    const int max_iters = args["iters"].as<int>();
    const int total = n * n;

    std::vector<double> a(total);
    std::vector<double> anew(total);
    init_grid(a, anew, n);

    double* grid = a.data();
    double* next = anew.data();
    double error = 1.0;
    int iter = 0;

    auto start = std::chrono::steady_clock::now();

#pragma acc data copy(grid[0:total], next[0:total])
    {
        while (error > eps && iter < max_iters) {
            error = 0.0;

#pragma acc parallel loop collapse(2) reduction(max:error) present(grid[0:total], next[0:total])
            for (int i = 1; i < n - 1; i++) {
                for (int j = 1; j < n - 1; j++) {
                    int k = i * n + j;
                    next[k] = 0.25 * (
                        grid[k - n] +
                        grid[k + n] +
                        grid[k - 1] +
                        grid[k + 1]
                    );
                    double diff = next[k] - grid[k];
                    if (diff < 0.0) {
                        diff = -diff;
                    }
                    if (diff > error) {
                        error = diff;
                    }
                }
            }

#pragma acc parallel loop collapse(2) present(grid[0:total], next[0:total])
            for (int i = 1; i < n - 1; i++) {
                for (int j = 1; j < n - 1; j++) {
                    int k = i * n + j;
                    grid[k] = next[k];
                }
            }

            iter++;
        }
    }

    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    std::cout << "size=" << n << "\n";
    std::cout << "iterations=" << iter << "\n";
    std::cout << "error=" << error << "\n";
    std::cout << "time=" << elapsed.count() << "\n";

    print_grid(a, n);
    return 0;
}
