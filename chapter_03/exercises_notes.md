# Chapter 3: Multidimensional Grids & Matrix Addition

This chapter covers the transition from 1D linear arrays to 2D matrix processing, focusing heavily on how thread coordinates map to physical memory and the performance implications of those mappings.

## 1. 1D to 2D Coordinate Mapping
When processing a 2D matrix using a purely 1D CUDA grid configuration, we must mathematically map the global 1D thread ID to a 2D `(row, col)` coordinate.

The global thread ID is calculated as:
$$ tid = blockIdx.x \cdot blockDim.x + threadIdx.x $$

Because C/C++ stores matrices in **Row-Major Order**, the width of the matrix (the number of columns) dictates the memory boundaries. We use integer division and modulo arithmetic against the column count to slice the 1D ID into 2D:

* **Row Index:** $tid / \text{numCols}$ (Truncates the decimal to find the current full row)
* **Column Index:** $tid \% \text{numCols}$ (Calculates the remainder to find the position within that row)

> **Important:** The total number of rows (`numRows`) is strictly used for boundary checking (`if (row < numRows)`), not for coordinate math.

---

## 2. Memory Coalescing vs. Strided Access
The way threads access physical RAM drastically impacts execution speed. The GPU executes threads in batches of 32, known as a **Warp**. 

When a warp requests memory, the hardware memory controller fetches data in contiguous 128-byte **Cache Lines**.

### Coalesced Access (Fast)
Assigning consecutive threads to consecutive columns in the same row results in **Coalesced Access**.
* The 32 threads in the warp request sequential addresses (e.g., indices `0` through `31`).
* These 32 floats (4 bytes each) fit perfectly into a single 128-byte cache line.
* **Hardware Cost:** 1 memory transaction.

### Strided Access (Slow / Anti-Pattern)
Assigning consecutive threads to consecutive rows in the same column results in **Strided Access**.
* The 32 threads request addresses separated by the matrix width (e.g., indices `0, 10000, 20000...`).
* The memory controller must fetch a completely different 128-byte cache line for every single thread, discarding the unused bytes.
* **Hardware Cost:** 32 memory transactions, saturating the VRAM bandwidth.

---

## 3. Kernel Implementations

### The Optimal Kernel (Row-Major / Coalesced)
```cpp
__global__ void matrixAddCoalesced(float *A, const float *B, const float *C, int numRows, int numCols) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = numRows * numCols;
    
    if (tid < totalElements) {
        // Map 1D to 2D: X-axis acts as Columns
        int row = tid / numCols; 
        int col = tid % numCols; 
        
        int idx = row * numCols + col; // Flattens back to tid
        A[idx] = B[idx] + C[idx];
    }
}