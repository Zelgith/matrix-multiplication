#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Matrix = std::vector<float>;

// Each CPU iteration computes one output element.
void cpu_seq(const Matrix& a, const Matrix& b, Matrix& c, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0;
            for (int k = 0; k < n; ++k)
                sum += a[i*n+k] * b[k*n+j];
            c[i*n+j] = sum;
        }
}

// Each thread gets separate rows. Join all threads before returning.
void cpu_parallel(const Matrix& a, const Matrix& b, Matrix& c, int n, int thread_count) {
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    try {
        for (int t = 0; t < thread_count; ++t) {
            int begin = n*t / thread_count;
            int end = n*(t+1) / thread_count;
            workers.emplace_back([&, begin, end] {
                for (int i = begin; i < end; ++i)
                    for (int j = 0; j < n; ++j) {
                        float sum = 0;
                        for (int k = 0; k < n; ++k)
                            sum += a[i*n+k] * b[k*n+j];
                        c[i*n+j] = sum;
                    }
            });
        }
    } catch (...) {
        for (auto& worker : workers) worker.join();
        throw;
    }
    for (auto& worker : workers) worker.join();
}

// One GPU thread computes one output element.
__global__ void gpu_kernel(const float* a, const float* b, float* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n*n) return;
    int i = index / n;
    int j = index % n;
    float sum = 0;
    for (int k = 0; k < n; ++k)
        sum += a[i*n+k] * b[k*n+j];
    c[index] = sum;
}

void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void gpu(const Matrix& a, const Matrix& b, Matrix& c, int n) {
    int count = n*n;
    std::size_t bytes = count * sizeof(float);
    float* memory = nullptr;
    cuda_check(cudaMalloc(&memory, 3*bytes));
    float* da = memory;
    float* db = memory + count;
    float* dc = memory + 2*count;
    try {
        cuda_check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
        gpu_kernel<<<(count+255)/256, 256>>>(da, db, dc, n);
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
        cuda_check(cudaDeviceSynchronize());
    } catch (...) {
        cudaFree(memory);
        throw;
    }
    cuda_check(cudaFree(memory));
}

void check_result(const Matrix& expected, const Matrix& actual) {
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i]-expected[i]) > 0.001f + 0.0001f*std::abs(expected[i]))
            throw std::runtime_error("Incorrect result at element " + std::to_string(i));
}

void generate(Matrix& a, Matrix& b) {
    std::srand(42);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = 2.0f * std::rand() / RAND_MAX - 1.0f;
        b[i] = 2.0f * std::rand() / RAND_MAX - 1.0f;
    }
}

void test_case(const char* name, const Matrix& a, const Matrix& b,
               const Matrix& expected, int n) {
    Matrix c(n*n, NAN);
    cpu_seq(a, b, c, n);
    check_result(expected, c);
    for (int threads : {1, 2, 4, 8}) {
        c.assign(n*n, NAN);
        cpu_parallel(a, b, c, n, threads);
        check_result(expected, c);
    }
    c.assign(n*n, NAN);
    gpu(a, b, c, n);
    check_result(expected, c);
    std::cout << "PASS: " << name << ", N=" << n
              << " (CPU, std::thread 1/2/4/8 threads, GPU)\n";
}

void tests() {
    test_case("known product", {1,2,3,4}, {5,6,7,8}, {19,22,43,50}, 2);
    test_case("fractional and negative", {-0.5f}, {2}, {-1}, 1);
    test_case("zero matrix", Matrix(9,0), Matrix(9,2), Matrix(9,0), 3);
    for (int n : {17,33}) {
        test_case("ones", Matrix(n*n,1), Matrix(n*n,1), Matrix(n*n,float(n)), n);
        Matrix a(n*n), b(n*n), expected(n*n), identity(n*n,0);
        generate(a, b);
        for (int i = 0; i < n; ++i) identity[i*n+i] = 1;
        test_case("identity", a, identity, a, n);
        cpu_seq(a, b, expected, n);
        test_case("random", a, b, expected, n);
    }
}

void measure(const std::string& mode, const Matrix& a, const Matrix& b,
             const Matrix& expected, int n, int threads, int repeats, int warmup_ms) {
    Matrix c(n*n);
    auto multiply = [&] {
        if (mode == "seq") cpu_seq(a, b, c, n);
        else if (mode == "par") cpu_parallel(a, b, c, n, threads);
        else gpu(a, b, c, n);
    };
    auto warmup_start = std::chrono::steady_clock::now();
    do {
        multiply();
        check_result(expected, c);
    } while (std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now()-warmup_start).count() < warmup_ms);

    for (int repeat = 1; repeat <= repeats; ++repeat) {
        auto start = std::chrono::steady_clock::now();
        multiply();
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now()-start).count();
        check_result(expected, c);
        std::cout << mode << ',' << n << ','
                  << (mode == "par" ? threads : mode == "seq" ? 1 : 0)
                  << ',' << repeat << ',' << ms << ",1\n";
    }
}

void benchmark(const std::string& mode, int n, int threads, int repeats, int warmup_ms) {
    Matrix a(n*n), b(n*n), expected(n*n);
    generate(a, b);
    cpu_seq(a, b, expected, n);
    if (mode == "all") {
        measure("seq", a, b, expected, n, 1, repeats, warmup_ms);
        for (int count : {1, 2, 4, 8})
            measure("par", a, b, expected, n, count, repeats, warmup_ms);
        measure("gpu", a, b, expected, n, 0, repeats, warmup_ms);
    } else {
        measure(mode, a, b, expected, n, threads, repeats, warmup_ms);
    }
}

int integer(const char* text) {
    std::size_t end = 0;
    int value = std::stoi(text, &end);
    if (text[end] != '\0') throw std::runtime_error("Invalid integer");
    return value;
}

int main(int argc, char** argv) {
    try {
        int repeats = 5;
        int warmup_ms = 0;
        std::vector<char*> arguments{argv[0]};
        for (int i = 1; i < argc; ++i) {
            std::string option = argv[i];
            if (option == "--repeats" || option == "--warmup-ms") {
                if (++i == argc) throw std::runtime_error("Missing value for " + option);
                int value = integer(argv[i]);
                if (option == "--repeats") repeats = value;
                else warmup_ms = value;
            } else {
                arguments.push_back(argv[i]);
            }
        }
        argc = static_cast<int>(arguments.size());
        argv = arguments.data();
        if (repeats < 1 || repeats > 1000 || warmup_ms < 0 || warmup_ms > 10000)
            throw std::runtime_error("repeats: 1..1000; warmup-ms: 0..10000");
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
            std::cout << "matmul seq|par|gpu [N=256] [threads=8]\n"
                         "matmul test\n"
                         "matmul all\n"
                         "all: sizes 64/128/256/512, CPU threads 1/2/4/8.\n"
                         "Options: --repeats 5 --warmup-ms 0\n"
                         "Warm-up: at least one call and the requested duration.\n"
                         "Defaults: one warm-up and five measurements. CSV on stdout.\n";
            return 0;
        }
        std::string mode = argv[1];
        if (mode == "test" && argc == 2) {
            tests();
            return 0;
        }
        bool all = mode == "all" && argc == 2;
        if (!all && (argc > 4 || (mode != "seq" && mode != "par" && mode != "gpu")))
            throw std::runtime_error("Use: matmul test | all | seq|par|gpu [N] [threads]");
        int n = argc > 2 ? integer(argv[2]) : 256;
        int threads = argc > 3 ? integer(argv[3]) : 8;
        if (n < 1 || n > 4096 || threads < 1 || threads > 128)
            throw std::runtime_error("N: 1..4096; threads: 1..128");
        std::cout << "implementation,n,threads,repeat,ms,ok\n";
        std::cout.precision(9);
        if (all)
            for (int size : {64, 128, 256, 512})
                benchmark("all", size, 8, repeats, warmup_ms);
        else
            benchmark(mode, n, threads, repeats, warmup_ms);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
