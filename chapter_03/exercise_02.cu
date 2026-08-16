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

__global__
void matrixVectorMultThreadPerRow(float *A, const float *B, const float *C, int M, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            sum += B[tid * N + col] * C[col];
        }
        A[tid] = sum;
    }
}

__global__
void matrixVectorMultWarpPerRow(float *A, const float *B, const float *C, int M, int N) {
    constexpr int warpSize = 32;
    const int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int warpId = globalThreadId / warpSize;
    const int lane = globalThreadId % warpSize;

    if (warpId < M) {
        float partialSum = 0.0f;
        for (int col = lane; col < N; col += warpSize) {
            partialSum += B[warpId * N + col] * C[col];
        }

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            partialSum += __shfl_down_sync(0xFFFFFFFF, partialSum, offset);
        }

        if (0 == lane) { A[warpId] = partialSum; }
    }
}

// CPU: Matrix-Vector Multiplication
void matrixVectorMultCPU(float *A, const float *B, const float *C, int M, int N) {
    for (int row = 0; row < M; row++) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            sum += B[row * N + col] * C[col];
        }
        A[row] = sum;
    }
}

bool verifyResults(const std::vector<float> &cpuResult, const std::vector<float> &gpuResult, float epsilon = 1e-4f) {
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

__host__
void runAndProfileKernels(int N) {
    const int totalElements = N * N;
    const size_t matSize = totalElements * sizeof(float);
    const size_t vecSize = N * sizeof(float);

    const std::vector<float> h_B(totalElements, 1.0f);
    const std::vector<float> h_C(N, 2.0f);

    // Allocate host memory for results
    std::vector<float> h_A_cpu(N, 0.0f);
    std::vector<float> h_A_gpu(N, 0.0f);

    float *d_A, *d_B, *d_C;
    cudaCheckErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), vecSize));
    cudaCheckErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), matSize));
    cudaCheckErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), vecSize));

    cudaCheckErrors(cudaMemcpy(d_B, h_B.data(), matSize, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_C, h_C.data(), vecSize, cudaMemcpyHostToDevice));

    // ---- Profile CPU Baseline ----
    // ---- Profile CPU Baseline ----
    std::cout << "Running CPU Baseline..." << std::endl;

    // 1. Warm-up run (primes the CPU cache and branch predictor)
    matrixVectorMultCPU(h_A_cpu.data(), h_B.data(), h_C.data(), N, N);

    const int cpuMonteCarloNumber = 100;
    float totalCpuMilliseconds = 0.0f;

    // 2. Profile over multiple iterations
    for (int i = 0; i < cpuMonteCarloNumber; i++) {
        // Clear the array to mimic the GPU's cudaMemset behavior for fairness
        std::fill(h_A_cpu.begin(), h_A_cpu.end(), 0.0f);

        auto cpu_start = std::chrono::high_resolution_clock::now();

        matrixVectorMultCPU(h_A_cpu.data(), h_B.data(), h_C.data(), N, N);

        auto cpu_stop = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float, std::milli> cpu_duration = cpu_stop - cpu_start;
        totalCpuMilliseconds += cpu_duration.count();
    }

    std::cout << "CPU Baseline Execution: " << totalCpuMilliseconds / cpuMonteCarloNumber
            << " ms (averaged over " << cpuMonteCarloNumber << " runs)\n" << std::endl;

    // ---- Profile GPU Kernels ----
    int threadsPerBlock = 256;
    int blocksPerGridRow = (N + threadsPerBlock - 1) / threadsPerBlock;
    int blockPerGridWarpPerRow = (N * 32 + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int totalMonteCarloNumber = 1e3;

    auto profile = [&](const char *label, auto launch) {
        float totalMilliseconds = 0.0f;
        for (int i = 0; i < totalMonteCarloNumber; i++) {
            cudaCheckErrors(cudaMemset(d_A, 0, vecSize));

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

    profile("Thread-per-row (uncoalesced)", [&]() {
        matrixVectorMultThreadPerRow<<<blocksPerGridRow, threadsPerBlock>>>(d_A, d_B, d_C, N, N);
    });

    cudaCheckErrors(cudaMemcpy(h_A_gpu.data(), d_A, vecSize, cudaMemcpyDeviceToHost));
    verifyResults(h_A_cpu, h_A_gpu);

    profile("Warp-per-row (coalesced)", [&]() {
        matrixVectorMultWarpPerRow<<<blockPerGridWarpPerRow, threadsPerBlock>>>(d_A, d_B, d_C, N, N);
    });

    cudaCheckErrors(cudaMemcpy(h_A_gpu.data(), d_A, vecSize, cudaMemcpyDeviceToHost));
    verifyResults(h_A_cpu, h_A_gpu);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 1e4;
    std::cout << "Testing Matrix-Vector Multiplication with N = " << N << " ..." << std::endl;
    runAndProfileKernels(N);
    return 0;
}
