#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>

#ifndef CLIENT_TASKS
#define CLIENT_TASKS 100
#endif

#ifndef TASK3_2_RESULTS_DIR
#define TASK3_2_RESULTS_DIR "."
#endif

static_assert(CLIENT_TASKS > 5 && CLIENT_TASKS < 10000, "CLIENT_TASKS must be in range 5 < N < 10000");

template <class T>
class TaskServer {
public:
    using task_type = std::function<T()>;

    void start() {
        std::lock_guard<std::mutex> lock(mut);

        if (server_thread.joinable()) {
            return;
        }

        stopped = false;
        server_thread = std::jthread([this](std::stop_token stoken) {
            work(stoken);
        });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mut);
            stopped = true;
        }

        cond_var.notify_all();

        if (server_thread.joinable()) {
            server_thread.request_stop();
            server_thread.join();
        }
    }

    size_t add_task(task_type task) {
        std::packaged_task<T()> package(std::move(task));
        std::future<T> future = package.get_future();

        std::lock_guard<std::mutex> lock(mut);

        size_t id = ++last_id;
        tasks.push({id, std::move(package), std::move(future)});

        cond_var.notify_one();
        return id;
    }

    T request_result(size_t id_res) {
        std::unique_lock<std::mutex> lock(mut);

        result_cond_var.wait(lock, [this, id_res] {
            return results.find(id_res) != results.end();
        });

        T value = results[id_res];
        results.erase(id_res);
        return value;
    }

    ~TaskServer() {
        stop();
    }

private:
    struct TaskRecord {
        size_t id = 0;
        std::packaged_task<T()> task;
        std::future<T> future;
    };

    void work(std::stop_token stoken) {
        while (true) {
            TaskRecord current;

            {
                std::unique_lock<std::mutex> lock(mut);

                cond_var.wait(lock, [this, &stoken] {
                    return stopped || stoken.stop_requested() || !tasks.empty();
                });

                if ((stopped || stoken.stop_requested()) && tasks.empty()) {
                    break;
                }

                current = std::move(tasks.front());
                tasks.pop();
            }

            current.task();
            T value = current.future.get();

            {
                std::lock_guard<std::mutex> lock(mut);
                results.insert({current.id, value});
            }

            result_cond_var.notify_all();
        }
    }

    std::queue<TaskRecord> tasks;
    std::unordered_map<size_t, T> results;

    std::mutex mut;
    std::condition_variable cond_var;
    std::condition_variable result_cond_var;

    std::jthread server_thread;
    bool stopped = true;
    size_t last_id = 0;
};

static double rand_double(std::mt19937& gen, double left, double right) {
    std::uniform_real_distribution<double> dist(left, right);
    return dist(gen);
}

static void prepare_file(std::ofstream& f) {
    f.setf(std::ios::fixed);
    f.precision(17);
    f << "id,operation,arg1,arg2,result\n";
}

static void client_sin(TaskServer<double>& server, int n, const std::filesystem::path& file_path) {
    const double pi = 3.14159265358979323846;
    std::mt19937 gen(101);
    std::ofstream f(file_path, std::ios::out | std::ios::trunc);
    prepare_file(f);

    for (int i = 0; i < n; i++) {
        double x = rand_double(gen, 0.0, 2.0 * pi);

        size_t id = server.add_task([x] {
            return std::sin(x);
        });

        double result = server.request_result(id);
        f << id << ",sin," << x << ",0," << result << "\n";
    }

    std::cout << "Saved CSV: " << file_path << std::endl;
}

static void client_sqrt(TaskServer<double>& server, int n, const std::filesystem::path& file_path) {
    std::mt19937 gen(202);
    std::ofstream f(file_path, std::ios::out | std::ios::trunc);
    prepare_file(f);

    for (int i = 0; i < n; i++) {
        double x = rand_double(gen, 0.0, 10000.0);

        size_t id = server.add_task([x] {
            return std::sqrt(x);
        });

        double result = server.request_result(id);
        f << id << ",sqrt," << x << ",0," << result << "\n";
    }

    std::cout << "Saved CSV: " << file_path << std::endl;
}

static void client_pow(TaskServer<double>& server, int n, const std::filesystem::path& file_path) {
    std::mt19937 gen(303);
    std::uniform_int_distribution<int> degree_dist(2, 6);
    std::ofstream f(file_path, std::ios::out | std::ios::trunc);
    prepare_file(f);

    for (int i = 0; i < n; i++) {
        double x = rand_double(gen, 0.5, 10.0);
        int degree = degree_dist(gen);

        size_t id = server.add_task([x, degree] {
            return std::pow(x, degree);
        });

        double result = server.request_result(id);
        f << id << ",pow," << x << "," << degree << "," << result << "\n";
    }

    std::cout << "Saved CSV: " << file_path << std::endl;
}

int main() {
    const int n = CLIENT_TASKS;

    std::filesystem::path results_dir = std::filesystem::path(TASK3_2_RESULTS_DIR);
    std::filesystem::create_directories(results_dir);

    TaskServer<double> server;
    server.start();

    std::cout << "Start" << std::endl;
    std::cout << "tasks_per_client=" << n << std::endl;

    std::thread client1(client_sin, std::ref(server), n, results_dir / "sin_results.csv");
    std::thread client2(client_sqrt, std::ref(server), n, results_dir / "sqrt_results.csv");
    std::thread client3(client_pow, std::ref(server), n, results_dir / "pow_results.csv");

    client1.join();
    client2.join();
    client3.join();

    server.stop();

    std::cout << "End" << std::endl;
    return 0;
}
