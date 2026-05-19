#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TASK3_2_RESULTS_DIR
#define TASK3_2_RESULTS_DIR "."
#endif

#ifndef CLIENT_TASKS
#define CLIENT_TASKS 100
#endif

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string item;

    while (std::getline(ss, item, ',')) {
        parts.push_back(item);
    }

    return parts;
}

static bool close_enough(double expected, double actual) {
    double scale = std::max(1.0, std::fabs(expected));
    return std::fabs(expected - actual) <= 1e-10 * scale;
}

static double expected_value(const std::string& operation, double arg1, double arg2) {
    if (operation == "sin") {
        return std::sin(arg1);
    }

    if (operation == "sqrt") {
        return std::sqrt(arg1);
    }

    return std::pow(arg1, arg2);
}

static bool check_file(const std::filesystem::path& file_path, const std::string& operation) {
    std::ifstream f(file_path);

    if (!f.is_open()) {
        std::cout << "File not found: " << file_path << std::endl;
        return false;
    }

    std::string line;
    std::getline(f, line);

    int checked = 0;
    int line_no = 1;
    bool ok = true;

    while (std::getline(f, line)) {
        line_no++;

        if (line.empty()) {
            continue;
        }

        std::vector<std::string> parts = split_csv(line);
        if (parts.size() != 5) {
            std::cout << "Bad line format in " << file_path << ":" << line_no << std::endl;
            ok = false;
            continue;
        }

        std::string cur_operation = parts[1];
        double arg1 = std::stod(parts[2]);
        double arg2 = std::stod(parts[3]);
        double result = std::stod(parts[4]);

        if (cur_operation != operation) {
            std::cout << "Bad operation in " << file_path << ":" << line_no << std::endl;
            ok = false;
            continue;
        }

        double expected = expected_value(cur_operation, arg1, arg2);
        if (!close_enough(expected, result)) {
            std::cout << "Wrong result in " << file_path << ":" << line_no
                      << " expected=" << expected
                      << " actual=" << result << std::endl;
            ok = false;
        }

        checked++;
    }

    if (checked != CLIENT_TASKS) {
        std::cout << "Bad row count in " << file_path
                  << " expected=" << CLIENT_TASKS
                  << " actual=" << checked << std::endl;
        ok = false;
    }

    std::cout << file_path.filename() << ": checked " << checked << " rows" << std::endl;
    return ok;
}

int main() {
    std::filesystem::path results_dir = std::filesystem::path(TASK3_2_RESULTS_DIR);

    bool ok = true;
    ok = check_file(results_dir / "sin_results.csv", "sin") && ok;
    ok = check_file(results_dir / "sqrt_results.csv", "sqrt") && ok;
    ok = check_file(results_dir / "pow_results.csv", "pow") && ok;

    if (ok) {
        std::cout << "Test passed" << std::endl;
        return 0;
    }

    std::cout << "Test failed" << std::endl;
    return 1;
}
