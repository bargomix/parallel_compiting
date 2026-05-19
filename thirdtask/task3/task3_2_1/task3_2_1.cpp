#include <algorithm>
#include <chrono>
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
#include <vector>

#ifndef CLIENT_TASKS
#define CLIENT_TASKS 100
#endif

#ifndef SERVER_WORKERS
#define SERVER_WORKERS 4
#endif

#ifndef TASK3_2_1_RESULTS_DIR
#define TASK3_2_1_RESULTS_DIR "."
#endif

static_assert(CLIENT_TASKS > 5 && CLIENT_TASKS < 10000, "CLIENT_TASKS must be in range 5 < N < 10000");
static_assert(SERVER_WORKERS > 0, "SERVER_WORKERS must be positive");

template <class T>
class TaskServer {
public:
    using task_type = std::function<T()>;

    explicit TaskServer(std::size_t workers_count = SERVER_WORKERS)
        : workers_count(std::max<std::size_t>(1, workers_count)) {
    }

    void start() {
        std::lock_guard<std::mutex> lock(mut);

        if (!workers.empty()) {
            return;
        }

        stopped = false;
        workers.reserve(workers_count);

        for (std::size_t i = 0; i < workers_count; i++) {
            workers.emplace_back([this](std::stop_token stoken) {
                work(stoken);
            });
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mut);
            stopped = true;
        }

        cond_var.notify_all();

        for (std::jthread& worker : workers) {
            if (worker.joinable()) {
                worker.request_stop();
            }
        }

        for (std::jthread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        workers.clear();
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

    std::size_t thread_count() const {
        return workers_count;
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

    mutable std::mutex mut;
    std::condition_variable cond_var;
    std::condition_variable result_cond_var;

    std::vector<std::jthread> workers;
    std::size_t workers_count = 1;
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

struct RunTimes {
    double init_s = 0.0;
    double work_s = 0.0;
    double stop_s = 0.0;
    double total_s = 0.0;
};

static void save_timing(const std::filesystem::path& results_dir,
                        const char* implementation,
                        int clients,
                        int tasks_per_client,
                        std::size_t server_workers,
                        const RunTimes& times) {
    const std::filesystem::path file_path = results_dir / "timing_results.csv";
    const bool need_header = !std::filesystem::exists(file_path) || std::filesystem::file_size(file_path) == 0;

    std::ofstream f(file_path, std::ios::out | std::ios::app);
    f.setf(std::ios::fixed);
    f.precision(9);

    if (need_header) {
        f << "implementation,clients,tasks_per_client,server_workers,init_time_s,work_time_s,stop_time_s,total_time_s\n";
    }

    f << implementation << ","
      << clients << ","
      << tasks_per_client << ","
      << server_workers << ","
      << times.init_s << ","
      << times.work_s << ","
      << times.stop_s << ","
      << times.total_s << "\n";
}

static void client_sin(TaskServer<double>& server, int n, const std::filesystem::path& file_path) {
    const double pi = 3.14159265358979323846;
    std::mt19937 gen(101);
    std::vector<std::pair<size_t, double>> tasks;
    tasks.reserve(n);

    for (int i = 0; i < n; i++) {
        double x = rand_double(gen, 0.0, 2.0 * pi);

        size_t id = server.add_task([x] {
            return std::sin(x);
        });

        tasks.push_back({id, x});
    }

    std::ofstream f(file_path, std::ios::out | std::ios::trunc);
    prepare_file(f);

    for (const auto& task : tasks) {
        double result = server.request_result(task.first);
        f << task.first << ",sin," << task.second << ",0," << result << "\n";
    }

    std::cout << "Saved CSV: " << file_path << std::endl;
}

static void client_sqrt(TaskServer<double>& server, int n, const std::filesystem::path& file_path) {
    std::mt19937 gen(202);
    std::vector<std::pair<size_t, double>> tasks;
    tasks.reserve(n);

    for (int i = 0; i < n; i++) {
        double x = rand_double(gen, 0.0, 10000.0);

        size_t id = server.add_task([x] {
            return std::sqrt(x);
        });

        tasks.push_back({id, x});
    }

    std::ofstream f(file_path, std::ios::out | std::ios::trunc);
    prepare_file(f);

    for (const auto& task : tasks) {
        double result = server.request_result(task.first);
        f << task.first << ",sqrt," << task.second << ",0," << result << "\n";
    }

    std::cout << "Saved CSV: " << file_path << std::endl;
}

static void client_pow(TaskServer<double>& server, int n, const std::filesystem::path& file_path) {
    struct TaskInfo {
        size_t id = 0;
        double x = 0.0;
        int degree = 0;
    };

    std::mt19937 gen(303);
    std::uniform_int_distribution<int> degree_dist(2, 6);
    std::vector<TaskInfo> tasks;
    tasks.reserve(n);

    for (int i = 0; i < n; i++) {
        double x = rand_double(gen, 0.5, 10.0);
        int degree = degree_dist(gen);

        size_t id = server.add_task([x, degree] {
            return std::pow(x, degree);
        });

        tasks.push_back({id, x, degree});
    }

    std::ofstream f(file_path, std::ios::out | std::ios::trunc);
    prepare_file(f);

    for (const TaskInfo& task : tasks) {
        double result = server.request_result(task.id);
        f << task.id << ",pow," << task.x << "," << task.degree << "," << result << "\n";
    }

    std::cout << "Saved CSV: " << file_path << std::endl;
}

int main() {
    using clock = std::chrono::steady_clock;

    const int n = CLIENT_TASKS;
    const int clients = 3;

    std::filesystem::path results_dir = std::filesystem::path(TASK3_2_1_RESULTS_DIR);
    std::filesystem::create_directories(results_dir);

    TaskServer<double> server(SERVER_WORKERS);
    const std::size_t server_workers = server.thread_count();

    std::cout.setf(std::ios::fixed);
    std::cout.precision(9);

    std::cout << "Start" << std::endl;
    std::cout << "tasks_per_client=" << n << std::endl;
    std::cout << "server_workers=" << server_workers << std::endl;

    auto total_start = clock::now();

    auto init_start = clock::now();
    server.start();
    auto init_end = clock::now();

    auto work_start = clock::now();
    std::thread client1(client_sin, std::ref(server), n, results_dir / "sin_results.csv");
    std::thread client2(client_sqrt, std::ref(server), n, results_dir / "sqrt_results.csv");
    std::thread client3(client_pow, std::ref(server), n, results_dir / "pow_results.csv");

    client1.join();
    client2.join();
    client3.join();
    auto work_end = clock::now();

    auto stop_start = clock::now();
    server.stop();
    auto stop_end = clock::now();

    auto total_end = clock::now();

    RunTimes times{
        std::chrono::duration<double>(init_end - init_start).count(),
        std::chrono::duration<double>(work_end - work_start).count(),
        std::chrono::duration<double>(stop_end - stop_start).count(),
        std::chrono::duration<double>(total_end - total_start).count()
    };

    std::cout << "init_time=" << times.init_s << std::endl;
    std::cout << "work_time=" << times.work_s << std::endl;
    std::cout << "stop_time=" << times.stop_s << std::endl;
    std::cout << "total_time=" << times.total_s << std::endl;

    save_timing(results_dir, "task3_2_1", clients, n, server_workers, times);
    std::cout << "Saved CSV: " << (results_dir / "timing_results.csv") << std::endl;

    std::cout << "End" << std::endl;
    return 0;
}
