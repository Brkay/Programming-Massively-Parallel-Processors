#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>

#define cudaCheckErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(const cudaError_t code, const char *file, const int line, const bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert:" << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

// VERSION 1: ONE THREAD PER ELEMENT, ROW-MAJOR (COALESCED)

// Consecutive threads calculate consecutive columns in the same row -> consecutive threads touch consecutive addresses.
// Matrix is assumed to be N x N.

__global__
void matrixAddElementCoalesced(float *A, const float *B, const float *C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N * N) {
        int row = tid / N;
        int col = tid % N;
        int idx = row * N + col; // == tid, kept for clarity

        A[idx] = B[idx] + C[idx];
    }
}

// VERSION 2: ONE THREAD PER ELEMENT, COL-MAJOR (STRIDED/SLOW)

// Consecutive threads calculate consecutive rows in the same column -> consecutive threads are N floats apart in memory.
// Matrix is assumed to be N x N.

__global__
void matrixAddElementStrided(float *A, const float *B, const float *C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N * N) {
        int col = tid / N;
        int row = tid % N;
        int idx = row * N + col;
        A[idx] = B[idx] + C[idx];
    }
}

// VERSION 3: ONE THREAD PER ROW (M X N) matrix

// Thread 'tid' owns row 'tid' and walks across all N columns.
// Bound check and loop bound are both in terms of the row's own length (N), independent of how many rows there are (M).

__global__
void matrixAddThreadPerRow(float *A, const float *B, const float *C, int M, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M) {
        int idx = tid * N;
        int upperIndex = idx + N; // first index of the NEXT row
        while (idx < upperIndex) {
            A[idx] = B[idx] + C[idx];
            ++idx;
        }
    }
}

// VERSION 4: ONE THREAD PER COLUMN (N X K) matrix.

// Thread 'tid' owns column 'tid' and walks down all N rows striding by K (the row length) between consecutive elements.

__global__
void matrixAddThreadPerColumn(float *A, const float *B, const float *C, int N, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < K) {
        int idx = tid;
        int upperIndex = tid + K * (N - 1); // Last row's index that corresponds to column (tid)

        while (idx <= upperIndex) {
            A[idx] = B[idx] + C[idx];
            idx += K;
        }
    }
}

// CPU: Matrix Addition (Flattened 1D array)
void matrixAddCPU(float *A, const float *B, const float *C, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        A[i] = B[i] + C[i];
    }
}

bool verifyResults(const std::vector<float>& cpuResult, const std::vector<float>& gpuResult, float epsilon = 1e-4f) {
    if (cpuResult.size() != gpuResult.size()) {
        std::cerr << "Size mismatch! CPU: " << cpuResult.size() << " GPU: " << gpuResult.size() << "\n";
        return false;
    }

    for (size_t i = 0; i < cpuResult.size(); i++) {
        if (std::abs(cpuResult[i] - gpuResult[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << "! "
                      << "CPU: " << cpuResult[i] << " vs GPU: " << gpuResult[i]
                      << " (Diff: " << std::abs(cpuResult[i] - gpuResult[i]) << ")\n";
            return false;
        }
    }
    std::cout << "Correctness Check: PASSED!\n";
    return true;
}

// HOST LAUNCHER & PROFILER (square N x N case)
__host__
void runAndProfileKernels(int N) {
    const int totalElements = N * N;
    const size_t size = totalElements * sizeof(float);

    const std::vector<float> h_B(totalElements, 1.0f);
    const std::vector<float> h_C(totalElements, 2.0f);

    // Allocate host memory for results
    std::vector<float> h_A_cpu(totalElements, 0.0f);
    std::vector<float> h_A_gpu(totalElements, 0.0f);

    float *d_A, *d_B, *d_C;
    cudaCheckErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), size));
    cudaCheckErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), size));
    cudaCheckErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), size));

    cudaCheckErrors(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_C, h_C.data(), size, cudaMemcpyHostToDevice));

    // ---- Profile CPU Baseline ----
    std::cout << "Running CPU Baseline..." << std::endl;

    // Warm-up run
    matrixAddCPU(h_A_cpu.data(), h_B.data(), h_C.data(), N, N);

    // Using fewer iterations than GPU because 100M elements is computationally heavy on CPU
    const int cpuMonteCarloNumber = 10;
    float totalCpuMilliseconds = 0.0f;

    for (int i = 0; i < cpuMonteCarloNumber; i++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        matrixAddCPU(h_A_cpu.data(), h_B.data(), h_C.data(), N, N);
        auto cpu_stop = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float, std::milli> cpu_duration = cpu_stop - cpu_start;
        totalCpuMilliseconds += cpu_duration.count();
    }
    std::cout << "CPU Baseline Execution: " << totalCpuMilliseconds / cpuMonteCarloNumber
              << " ms (averaged over " << cpuMonteCarloNumber << " runs)\n" << std::endl;


    // ---- Profile GPU Kernels ----
    int threadsPerBlock = 256;

    // Grid size for 1 thread per element (Total Threads = N*N)
    int blocksPerGridElement = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    // Grid size for 1 thread per row OR column (Total Threads = N)
    int blocksPerGridRowCol = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 100 iterations is plenty for a highly memory-bound operation like Matrix Addition
    const int totalMonteCarloNumber = 100;

    auto profile = [&](const char *label, auto launch) {
        float totalMilliseconds = 0.0f;
        for (int i = 0; i < totalMonteCarloNumber; i++) {
            cudaCheckErrors(cudaMemset(d_A, 0, size));

            cudaEventRecord(start);
            launch();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            totalMilliseconds += ms;
        }
        std::cout << label << ":" << totalMilliseconds / totalMonteCarloNumber << " ms" << std::endl;
    };

    profile("1 Thread/Element (Row-Major/Coalesced)", [&]() {
        matrixAddElementCoalesced<<<blocksPerGridElement, threadsPerBlock>>>(d_A, d_B, d_C, N);
    });

    // ---- Correctness Verification ----
    cudaCheckErrors(cudaMemcpy(h_A_gpu.data(), d_A, size, cudaMemcpyDeviceToHost));
    verifyResults(h_A_cpu, h_A_gpu);
    profile("1 Thread/Element (Col-Major/Strided)", [&]() {
        matrixAddElementStrided<<<blocksPerGridElement, threadsPerBlock>>>(d_A, d_B, d_C, N);
    });
    // ---- Correctness Verification ----
    cudaCheckErrors(cudaMemcpy(h_A_gpu.data(), d_A, size, cudaMemcpyDeviceToHost));
    verifyResults(h_A_cpu, h_A_gpu);
    profile("1 Thread/Row (Strided)", [&]() {
        matrixAddThreadPerRow<<<blocksPerGridRowCol, threadsPerBlock>>>(d_A, d_B, d_C, N, N);
    });
    // ---- Correctness Verification ----
    cudaCheckErrors(cudaMemcpy(h_A_gpu.data(), d_A, size, cudaMemcpyDeviceToHost));
    verifyResults(h_A_cpu, h_A_gpu);
    profile("1 Thread/Col (Coalesced)", [&]() {
        matrixAddThreadPerColumn<<<blocksPerGridRowCol, threadsPerBlock>>>(d_A, d_B, d_C, N, N);
    });

    // ---- Correctness Verification ----
    cudaCheckErrors(cudaMemcpy(h_A_gpu.data(), d_A, size, cudaMemcpyDeviceToHost));
    verifyResults(h_A_cpu, h_A_gpu);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 1e4;
    std::cout << "Testing Matrix Addition with N = " << N << " (" << N*N << " elements)..." << std::endl;
    runAndProfileKernels(N);
    return 0;
}