#include <array>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

// Macro to catch silent hardware or launch failures
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}

// dimX = Total Columns (Width), dimY = Total Rows (Height)
__global__
void matrixAddKernel_b(float *A, const float *B, const float *C, int dimX, int dimY) {

    int col = threadIdx.x + blockIdx.x * blockDim.x; // X handles Columns
    int row = threadIdx.y + blockIdx.y * blockDim.y; // Y handles Rows

    // 1. FIXED: col checked against dimX, row checked against dimY
    if (col < dimX && row < dimY) {

        // 2. FIXED: Multiply row by Total Columns (dimX)
        int idx = row * dimX + col;

        A[idx] = C[idx] + B[idx];
    }
}

__host__
void matrixAdd(float *A, const float *B, const float *C, const std::array<int, 2>& dim) {
    const int size = dim[0] * dim[1] * sizeof(float);
    float *d_A, *d_B, *d_C;

    // 3. ADDED: Error checking to catch missing GPU or out-of-memory errors
    cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&d_C), size));
    cudaCheckError(cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&d_B), size));
    cudaCheckError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&d_A), size));

    // Define a 16x16 block
    dim3 dimBlock(16, 16, 1);

    // dim[0] is X (Columns), dim[1] is Y (Rows)
    dim3 dimGrid((dim[0] + dimBlock.x - 1) / dimBlock.x,
                 (dim[1] + dimBlock.y - 1) / dimBlock.y,
                 1);

    // Launch the kernel
    matrixAddKernel_b<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, dim[0], dim[1]);

    // Catch kernel execution errors (e.g., too many threads per block)
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back
    cudaCheckError(cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // dim[0] = Width (Columns), dim[1] = Height (Rows)
    constexpr std::array<int, 2> dim{256, 256};

    constexpr int totalElements = dim[0] * dim[1];
    std::vector<float> A(totalElements, 0.0f); // Output buffer
    std::vector<float> B(totalElements, 2.0f);
    std::vector<float> C(totalElements, 2.0f);

    matrixAdd(A.data(), B.data(), C.data(), dim);

    // Verify the result: Only print the first element to avoid flooding the terminal
    std::cout << "Element A[0] = " << A[0] << " (Expected 4)" << std::endl;

    return 0;
}