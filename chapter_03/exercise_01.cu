#include <vector>
#include <iostream>
#include <cuda_runtime.h>


#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

// ---------------------------------------------------------
// VERSION 1: COALESCED (FAST)
// Consecutive threads calculate consecutive columns in the same row.
// ---------------------------------------------------------
__global__
void matrixAddCoalesced(float *A, const float *B, const float *C, int N) {
    // 1D Thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N * N) {
        // Map 1D thread to 2D matrix: X-axis acts as Columns
        int row = tid / N;
        int col = tid % N;

        // Flatten back to 1D index (Notice this just equals 'tid')
        int idx = row * N + col;

        A[idx] = B[idx] + C[idx];
    }
}

// ---------------------------------------------------------
// VERSION 2: STRIDED (SLOW)
// Consecutive threads calculate consecutive rows in the same column.
// ---------------------------------------------------------
__global__
void matrixAddStrided(float *A, const float *B, const float *C, int N) {
    // 1D Thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N * N) {
        // Map 1D thread to 2D matrix: X-axis acts as Rows (Flipped!)
        int col = tid / N;
        int row = tid % N;

        // Flatten back to 1D index
        // Thread 0 hits index 0. Thread 1 hits index N. Thread 2 hits 2N.
        int idx = row * N + col;

        A[idx] = B[idx] + C[idx];
    }
}

// ---------------------------------------------------------
// HOST LAUNCHER & PROFILER
// ---------------------------------------------------------
__host__
void runAndProfileKernels(int N) {
    const int totalElements = N * N;
    const size_t size = totalElements * sizeof(float);

    std::vector<float> h_B(totalElements, 1.0f);
    std::vector<float> h_C(totalElements, 2.0f);
    std::vector<float> h_A(totalElements, 0.0f);

    float *d_A, *d_B, *d_C;
    cudaCheckError(cudaMalloc((void **)&d_A, size));
    cudaCheckError(cudaMalloc((void **)&d_B, size));
    cudaCheckError(cudaMalloc((void **)&d_C, size));

    cudaCheckError(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_C, h_C.data(), size, cudaMemcpyHostToDevice));

    // 1D Hardware Configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for precise hardware timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int totalMonteCarloNumber = 1e3;
    float milliseconds = 0;

    // --- Profile Coalesced ---
    float totalMilliseconds = 0;
    for (int i = 0; i < totalMonteCarloNumber; i++) {
        cudaEventRecord(start);
        matrixAddCoalesced<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalMilliseconds += milliseconds;
    }


    std::cout << "Coalesced (Row-Major) Time: " << totalMilliseconds / totalMonteCarloNumber << " ms\n";

    // --- Profile Strided ---
    // Clear the output buffer to ensure a fair test
    cudaMemset(d_A, 0, size);

    totalMilliseconds = 0;
    for (int i = 0; i < totalMonteCarloNumber; i++) {
        cudaEventRecord(start);
        matrixAddStrided<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalMilliseconds += milliseconds;
    }

    std::cout << "Strided (Col-Major) Time:   " << totalMilliseconds / totalMonteCarloNumber << " ms\n";

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // A 10,000 x 10,000 matrix creates 100 Million threads.
    // This size is large enough to force the GPU to work hard,
    // making the memory traffic bottleneck very obvious.
    int N = 10000;

    std::cout << "Testing Matrix Addition with N = " << N << "...\n";
    runAndProfileKernels(N);

    return 0;
}
