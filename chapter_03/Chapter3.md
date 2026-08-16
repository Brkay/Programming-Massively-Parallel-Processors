# PMPP Chapter 3 — CUDA Kernel Exercises

## Q3.1: Matrix Addition
**Question:** Write a kernel that adds two matrices, `A = B + C`, handling the general `M x N` (non-square) case. Show different ways of mapping CUDA threads onto matrix elements: 
1. One thread per output element (row-major/coalesced and col-major/strided).
2. One thread per row.
3. One thread per column.

**Answer:**

**1. One thread per element — row-major (coalesced)**
Thread `tid` maps directly onto a flattened index. Consecutive threads touch consecutive addresses.
```cpp
__global__
void matrixAddElementCoalesced(float *A, const float *B, const float *C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N * N) {
        int row = tid / N;
        int col = tid % N;
        int idx = row * N + col;
        A[idx] = B[idx] + C[idx];
    }
}
```

**2. One thread per element — col-major (strided, unoptimized)**
`row` and `col` are swapped. Consecutive threads land `N` floats apart, breaking coalescing.
```cpp
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
```

**3. One thread per row (general `M x N`)**
Thread `tid` owns row `tid` and iterates across its `N` columns.
```cpp
__global__
void matrixAddThreadPerRow(float *A, const float *B, const float *C, int M, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M) {
        int idx = tid * N;
        int upperIndex = idx + N;
        while (idx < upperIndex) {
            A[idx] = B[idx] + C[idx];
            ++idx;
        }
    }
}
```

**4. One thread per column (general `N x K`, N rows / K cols)**
Thread `tid` owns column `tid`. Elements of a column are `K` (row length) apart.
```cpp
__global__
void matrixAddThreadPerColumn(float *A, const float *B, const float *C, int N, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < K) {
        int idx = tid;
        int upperIndex = tid + K * (N - 1);
        while (idx <= upperIndex) {
            A[idx] = B[idx] + C[idx];
            idx += K;
        }
    }
}
```

### Benchmark Environment & Configuration

**Hardware**
* **GPU:** NVIDIA GeForce RTX 4080 (16GB GDDR6X, Compute Capability 8.9)
* **CPU:** AMD Ryzen 7 7800X3D (96MB L3 Cache)
* **RAM:** 32GB Corsair DDR5-6000 (2x16GB, EXPO Enabled)
* **Host System:** Ubuntu 24.04.4 LTS (Kernel 6.8.0-137-generic)

**Software & Toolchain**
* **IDE & Build System:** JetBrains CLion (CMake 3.20+)
* **NVIDIA Driver:** 610.57.04
* **CUDA Version:** 13.3
* **Language Standards:** C++17, CUDA C++17
* **Target Architecture:** `sm_89` (Ada Lovelace)

**Build Configurations Tested (via CMake/CLion)**
* **Release (Aggressive Optimization):**
  * *Build Type:* `Release`
  * *Host Flags:* `-O3 -march=native -DNDEBUG`
  * *Device Flags:* `-O3 -use_fast_math --ptxas-options=-v`
* **Debug (Unoptimized):**
  * *Build Type:* `Debug`
  * *Host Flags:* `-O0 -g -Wall -Wextra`
  * *Device Flags:* `-O0 -g -G -lineinfo`

**Workload Configuration**
* **Algorithm:** Matrix Addition (FP32), `A = B + C`
* **Dimensions:** N = 10,000 (100M elements per matrix, ~1.2GB total memory footprint for A, B, and C)
* **Block Size:** 256 threads
* **Measurement:** `std::chrono` (CPU, averaged over 10 runs) and CUDA Events (GPU, averaged over 100 runs). GPU correctness verified against CPU output within a `1e-4` epsilon tolerance.

---

### Benchmark Results (Matrix Addition, N = 10,000)

| Processor | Kernel Architecture | Build Profile | Execution Time (ms) |
| :--- | :--- | :--- | :--- |
| 🖥️ **CPU**| **Naive loops (1D flattened)** | Release | 28.2724 |
| 🟩 **GPU** | **1 Thread/Element (Row-Major/Coalesced)** | Release | 2.0332 |
| 🟩 **GPU** | **1 Thread/Element (Col-Major/Strided)** | Release | 6.8367 |
| 🟩 **GPU** | **1 Thread/Row (Strided)** | Release | 7.1471 |
| 🟩 **GPU** | **1 Thread/Col (Coalesced)** | Release | 3.7703 |
| 🖥️ **CPU**| **Naive loops (1D flattened)** | Debug | 128.8270 |
| 🟩 **GPU** | **1 Thread/Element (Row-Major/Coalesced)** | Debug | 2.3142 |
| 🟩 **GPU** | **1 Thread/Element (Col-Major/Strided)** | Debug | 7.7361 |
| 🟩 **GPU** | **1 Thread/Row (Strided)** | Debug | 7.8721 |
| 🟩 **GPU** | **1 Thread/Col (Coalesced)** | Debug | 8.7845 |

---

### Performance Analysis

#### 1. Baseline coalescing cost — 1 Thread/Element, row-major vs col-major

Comparing the two "1 Thread/Element" kernels in Release, the row-major (coalesced) layout is **~3.36x faster** (2.03 ms vs 6.83 ms) than the col-major (strided) layout. Matrix addition does one FP32 op per three memory accesses (two reads, one write) — it is entirely bandwidth-bound, so this gap is purely the cost of scattering a warp's 32 memory requests across 32 separate cache lines instead of packing them into one.

At 2.0332 ms for 100M elements, this kernel moves 1.2 GB (200M floats read + 100M floats written) in ~0.002 s → **~590 GB/s, ~82% of the RTX 4080's ~716 GB/s theoretical peak.** This is what "good" looks like when the grid is large enough to saturate the GPU — keep it as the reference point for the two kernels below, which never get close to it despite one of them being just as well-coalesced.

#### 2. Thread/Row vs Thread/Col — which one is actually coalesced

Coalescing is decided by what the **32 lanes of one warp** request **on the same instruction**, not by whether a single thread's own sequence of accesses looks orderly over time.

**Thread/row** — each thread owns a whole row and walks it with `++idx`:
```cpp
if (tid < M) {
    int idx = tid * N;
    while (idx < idx + N) { A[idx] = B[idx] + C[idx]; ++idx; }
}
```
At any fixed loop iteration, lane `l`'s address is `(base + l) * N + offset`. Adjacent lanes are `N` floats apart — for N = 10,000, addresses land 40,000 bytes away from each other. **Strided / uncoalesced.**

**Thread/col** — each thread owns a column and walks it with `idx += K`:
```cpp
if (tid < K) {
    int idx = tid;
    while (idx <= upperIndex) { A[idx] = B[idx] + C[idx]; idx += K; }
}
```
At any fixed loop iteration, lane `l`'s address is `(base + l) + offset * K`. Adjacent lanes are **1 float apart**. **Coalesced.**

**a) Sectors requested per warp instruction.** Global memory moves in 32-byte sectors; a warp's 32 contiguous floats fit exactly into 4 sectors (128 bytes) — that's the best case. 32 scattered floats need up to 32 separate sectors — the worst case. Run the below sh commands in the executable directory (debug and release directories are different in mine due to CLion configuration but it can be handled however you like in the CMake file) to get ncu analysis data.

```bash
ncu --kernel-name matrixAddThreadPerRow --launch-skip 5 --launch-count 1 \
    --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
    -o row_sectors \
    ./ch03_ex02

ncu --kernel-name matrixAddThreadPerColumn --launch-skip 5 --launch-count 1 \
    --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
    -o col_sectors \
    ./ch03_ex02

ncu --import row_sectors.ncu-rep
ncu --import col_sectors.ncu-rep
```

| Kernel | Sectors / request | Interpretation |
|---|---|---|
| Thread/Row | **31.95** | ≈32 — essentially the worst case: every lane needs its own sector |
| Thread/Col | **3.99** | ≈4 — essentially the ideal case: one transaction serves the whole warp |

**b) The `--set full` report's own diagnostic**, present verbatim in row's output and **absent** from col's in every build:

> *"The memory access pattern for global loads from L1TEX might not be optimal. On average, only 4.0 of the 32 bytes transmitted per sector are utilized by each thread... This workload has uncoalesced global accesses resulting in a total of 262,500,000 excessive sectors (88% of the total 300,000,000 sectors)."*

12.5% byte utilization, 88% wasted bandwidth — row, both builds. Col never triggers this warning in either build.

**c) L2 hit rate**, which stays essentially constant across build type (confirming it's a property of the access pattern, not the compiler):

| Kernel | L2 Hit Rate (Debug) | L2 Hit Rate (Release) |
|---|---|---|
| Thread/Row | 84.14% | 82.25% |
| Thread/Col | 32.42% | 32.61% |

Row's thread re-reads the same 128-byte line ~32 times in a row (sequential `++idx`) — high reuse. Col's thread jumps a full `K` floats (40,000 bytes) every iteration — a brand-new line every time, never revisited.

Commands used to gather the quick metrics table above (before the `--set full` dive):
```bash
ncu --kernel-name matrixAddThreadPerRow --launch-skip 5 --launch-count 1 \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,lts__t_sector_hit_rate.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct \
    -o release_row \
    ./ch03_ex02

ncu --kernel-name matrixAddThreadPerColumn --launch-skip 5 --launch-count 1 \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,lts__t_sector_hit_rate.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct \
    -o release_col \
    ./ch03_ex02

# repeat both against ./ch03_ex02 for debug_row / debug_col
```

#### 3. Release: why the coalesced kernel (col) still beats the uncoalesced one (row) by only ~1.9x, and why row's "cost location" is different from col's

Naively you might expect the uncoalesced kernel to be much worse than 1.9x slower, given it needs ~8x more sectors per request. It isn't, because a sector that's an L2 *hit* is cheap — row's 32 requests are individually fast, they're just numerous. The `--set full` "Speed Of Light" section shows this as two kernels bottlenecked on **different parts of the memory hierarchy**:

```bash
ncu --kernel-name matrixAddThreadPerRow    --launch-skip 5 --launch-count 1 --set full -o row_release_full ./ch03_ex02
ncu --kernel-name matrixAddThreadPerColumn --launch-skip 5 --launch-count 1 --set full -o col_release_full ./ch03_ex02
ncu --kernel-name matrixAddThreadPerRow    --launch-skip 5 --launch-count 1 --set full -o row_debug_full   ./ch03_ex02
ncu --kernel-name matrixAddThreadPerColumn --launch-skip 5 --launch-count 1 --set full -o col_debug_full   ./ch03_ex02

ncu --import row_release_full.ncu-rep
# ...repeat --import for the other three
```

`ncu` reports "Memory Throughput" as whichever sub-unit (L1TEX / L2 / DRAM) is closest to *its own* peak — that sub-unit is the bottleneck:

| | Row | Col |
|---|---|---|
| **Release** | Memory Throughput 49.21% = **L2 Cache Throughput** 49.21% (DRAM only 24.77%) → **L2-bound** | Memory Throughput 45.73% = **DRAM Throughput** 45.73% (L2 only 9.53%) → **DRAM-bound** |
| **Debug** | Memory Throughput 45.96% = **L2 Cache Throughput** 45.96% (DRAM only 21.24%) → **L2-bound** | Memory Throughput 18.78% = **DRAM Throughput** 18.78% (L2 only 4.01%) → **DRAM-bound** |

This classification is stable across both builds — another confirmation that it's a fixed property of the access pattern, not something the compiler changes. Row is drowning L2 in a flood of small transactions; col is going straight past L1/L2 to DRAM for a smaller number of larger round trips.

The "Warp State Statistics" section quantifies the resulting per-instruction cost directly, in cycles:

| | Row (Release) | Col (Release) |
|---|---|---|
| Warp cycles per instruction | **308.15** | **84.90** |
| ...of which long-scoreboard stall | 303.4 cyc (98.5%) | 79.3 cyc (93.4%) |

Row costs **~3.6x more cycles per warp instruction** than col, in the same direction and roughly the same magnitude as the wall-clock gap (7.15 ms vs 3.77 ms, ~1.9x — the two numbers aren't expected to match exactly, since "cycles per instruction" is averaged across *all* instructions in the kernel, including cheap non-memory ones, while wall-clock time reflects the full instruction stream end to end). The mechanism: even though most of row's 32 sector-fetches are individually fast L2 hits, the load-store unit has to issue and retire all 32 sequentially before the one warp instruction can retire — "death by a thousand fast cuts" outweighs col's "few slower round trips" once occupancy is too low (see §5) to overlap either one away.

#### 4. Debug: the order inverts — row becomes faster than col

From the benchmark table: row goes from 7.1471 ms (release) → 7.8721 ms (debug), a **+10.2%** slowdown. Col goes from 3.7703 ms → 8.7845 ms, a **+133.0%** slowdown — more than double. Col isn't recovering any advantage; it's being hit far harder by `-O0 -G` than row is, hard enough to fall behind.

The clearest evidence for *why* is in "Instruction Statistics":

| | Row | Col |
|---|---|---|
| Executed instructions, Release | 15,468,815 | 34,434,424 |
| Executed instructions, Debug | 187,820,691 | 187,821,317 |
| Debug / Release ratio | **12.14x** | **5.45x** |

Notice the two debug-mode instruction counts: **187,820,691 vs 187,821,317 — a difference of 626 instructions out of 187.8 million (0.0003%).** In release, row and col had visibly different instruction counts (15.5M vs 34.4M) because the compiler could strength-reduce row's simple `++idx` walk more aggressively than col's `idx += K` stride. Under `-O0`, that optimization gap disappears — both loop bodies are structurally near-identical (index update, two loads, one add, one store, a compare, a branch), and with essentially no optimization applied to either, the compiler emits almost exactly the same number of instructions for both kernels.

That has two consequences visible in "Warp Cycles Per Instruction":

| | Row | Col |
|---|---|---|
| Release | 308.15 cyc/instr | 84.90 cyc/instr |
| Debug | 32.79 cyc/instr | 38.56 cyc/instr |

Per-instruction cost *drops* for both in debug — but this isn't the memory system getting faster; it's the same average now diluted across ~12x (row) / ~5.4x (col) more total instructions, most of which are cheap address-arithmetic ops rather than memory ops. Because row's instruction count inflates so much more (its release baseline was artificially compressed by aggressive optimization), row's per-instruction average dilutes further than col's — enough to drop below col's, flipping the ranking. Debug L2 hit rates (row 84%, col 32%) and the SoL bound classification (row L2-bound, col DRAM-bound) are unchanged from release, confirming the underlying memory-access pattern itself never changes — only how much non-memory instruction volume surrounds it.

> **Confidence note:** the instruction-count and per-instruction-cycle evidence above reliably explains *that* debug hits col harder than row, and by roughly the right proportions. The precise reason the compiler's un-optimized code generation is so much more sensitive for col specifically is inferred rather than directly measured. To pin it down exactly, disassemble both kernels and compare instruction mix directly:
> ```bash
> nvdisasm ./ch03_ex02.cubin > debug_disasm.txt   # or extract via cuobjdump --dump-sass
> ```
> or pull a full stall-reason breakdown (`smsp__warp_issue_stalled_*`) rather than just long-scoreboard, to see whether the extra debug-mode instructions themselves are stalling on something (e.g. local-memory traffic) independent of the global-load pattern.

#### 5. Why occupancy is stuck at ~16% everywhere — and it has nothing to do with debug vs release

All four configurations land in the same narrow band:

| | Row Debug | Col Debug | Row Release | Col Release |
|---|---|---|---|---|
| Achieved Occupancy | 16.35% | 16.28% | 16.29% | 15.91% |
| Registers/thread | 18 | 19 | 20 | 16 |
| Block Limit (Registers) | 10 | 10 | 10 | 16 |
| Block Limit (Warps) | **6** | **6** | **6** | **6** |

`ncu` computes theoretical occupancy from the *minimum* of several independent per-SM ceilings (max blocks the hardware tracks, register budget, shared-memory budget, and a hard warp-count ceiling). On this GPU (Ada, CC 8.9), **an SM caps out at 48 resident warps, full stop** — that's `Block Limit Warps = 48 / 8 warps-per-block = 6`, a fixed hardware number unrelated to your code. In every one of the four runs, `6` is smaller than the register-based limit (10–16) — so register pressure, which *does* change meaningfully between builds (16→20), never actually binds. That's why occupancy barely moves despite debug mode visibly increasing register usage: the tighter ceiling was never the register one.

The real cause is in "Launch Statistics": **Grid Size = 40 blocks**, against **76 SMs** on the device. Even in the best case (one block per SM), 36 of 76 SMs get zero work — confirmed by `Waves Per SM: 0.09` (you're launching 9% of what it would take to fill the GPU even once) and by the "GPU and Memory Workload Distribution" section flagging some SM instances at "100.00% below average" (literally idle). `Achieved Active Warps per SM` lands at ~7.6–7.85 in every run — just under one block's worth of warps (8), averaged across 76 SMs including the idle ones.

Contrast this with the first benchmark in §1 (thread/element, 100M threads ≈ 390,625 blocks): that grid is large enough to give every SM many blocks across multiple waves, which is why it reaches ~590 GB/s (82% of hardware peak) while these row/col kernels — one of which is just as well-coalesced — never break 320 GB/s.

#### 6. Two independent axes

This dataset cleanly separates two things that are easy to conflate:

* **Coalescing = efficiency** — how much of the bytes you fetch do you actually use? Col: ~100%. Row: ~12.5% (only 4 of 32 bytes per sector).
* **Occupancy = saturation** — do you have enough concurrent warps to keep the memory pipeline full and hide DRAM latency? Both row and col are stuck at ~16% here, for the same grid-size reason, regardless of how well-coalesced either one is.

Col has near-perfect efficiency but is still latency-bound, because occupancy is too low to launch enough concurrent DRAM requests to hide their round-trip cost — which is exactly why it tops out at 129.70–315.80 GB/s here instead of anywhere near the 590 GB/s the identically-coalesced thread/element kernel reaches once its grid is large enough to actually saturate the GPU.

---

## Q3.2: Matrix-Vector Multiplication

**Question:** Write a kernel computing `A = B * C` where `B` is an `M x N` matrix and `C` is an `N x 1` vector, producing an `M x 1` result `A`. Compare a straightforward one-thread-per-row mapping against a memory-access-optimized version.

**Answer:**

**Baseline: One thread per row (Uncoalesced)**
Consecutive threads access elements separated by N floats, resulting in poorly coalesced global-memory accesses and potentially many memory transactions.

```cpp
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
```

**Optimized: One warp per row (Coalesced)**
Warp `w` owns row `w`, and its 32 lanes split up the row's columns. This ensures consecutive threads read consecutive floats, creating coalesced memory transactions. A warp-level tree reduction (`__shfl_down_sync`) is used to sum the partial results.

```cpp
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
```
---

**Workload Configuration**
* **Algorithm:** Matrix-Vector Multiplication (FP32)
* **Dimensions:** N = 10,000 (100M elements, ~400MB dataset)
* **Block Size:** 256 threads
* **Measurement:** `std::chrono` (CPU, averaged over 100 runs) and CUDA Events (GPU, averaged over 1,000 runs). GPU correctness verified against CPU output within a `1e-4` epsilon tolerance.

### Benchmark Results

| Processor | Kernel Architecture | Build Profile | Execution Time (ms) |
| :--- | :--- | :--- | :--- |
| 🖥️ **CPU**| **Naive loops (1D flattened)** | Release | 63.6714 |
| 🟩 **GPU** | **Thread-per-row (uncoalesced)** | Release | 1.4920 |
| 🟩 **GPU** | **Warp-per-row (coalesced)** | Release | 0.6046 |
| 🖥️ **CPU**| **Naive loops (1D flattened)** | Debug | 243.9990 |
| 🟩 **GPU** | **Thread-per-row (uncoalesced)** | Debug | 4.4354 |
| 🟩 **GPU** | **Warp-per-row (coalesced)** | Debug | 1.1388 |

### Performance Analysis

* **CPU vs. GPU Acceleration:** In the optimized Release build, the best GPU implementation (warp-per-row) drastically outperforms the CPU baseline, delivering a **~105x speedup** (63.67 ms vs 0.60 ms). This showcases the RTX 4080's massive memory bandwidth and parallel processing capabilities over the host processor for vectorizable workloads.
* **Memory Coalescing Impact:** Transitioning from an uncoalesced thread-per-row approach to a coalesced warp-per-row approach yields a **~2.47x speedup** (1.49 ms vs 0.60 ms) on the GPU. This validates that maximizing memory transaction efficiency is critical for bandwidth-bound operations like matrix-vector multiplication.
* **Compiler Optimization Overhead:** The Debug build introduces severe performance penalties across both processors due to disabled optimizations (`-O0`) and injected debug symbols. The CPU code runs **~3.8x slower**, while the GPU kernels run up to **~3x slower** in debug mode. This highlights why performance profiling must strictly be isolated to Release builds.

---

## Q3.3: Grid Sizing Reference
**Question:** How should the grid size be calculated for different thread mapping patterns?

**Answer:**
The grid must be sized based on what the kernel actually iterates over to prevent launching no-op threads or under-launching. 

| Kernel Pattern | Threads Needed | Grid Size Formula (`tpb` = threadsPerBlock) |
|---|---|---|
| One thread per element | `M * N` | `(M * N + tpb - 1) / tpb` |
| One thread per row | `M` | `(M + tpb - 1) / tpb` |
| One thread per column | `K` | `(K + tpb - 1) / tpb` |
| One warp per row | `M * 32` | `(M * 32 + tpb - 1) / tpb` |

---

## Q3.4: Host-Device Function Declarations
**Question:** Can a `__global__` function be called directly from host code like a standard function, and how do you declare a function to run on both the host and the device?

**Answer:**
No. A `__global__` function is a kernel—it runs on the device and can only be executed via a kernel launch using the `<<<...>>>` syntax. 

To create a shared helper function that compiles for both the CPU and GPU, use `__host__ __device__` together:
```cpp
__host__ __device__
float square(float x) { return x * x; }
```

---

## Q3.5: Allocate Device Memory and Kernel Launch Code
**Question:** Write the host code to allocate device memory, copy data from the host, launch a vector addition kernel, and copy the results back.

**Answer:**
```cpp
#include <cuda_runtime.h>

void vecAdd(float* A, float* B, float* C, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Allocate Device Memory
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // Copy Host to Device
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Kernel Launch
    vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    // Copy Device to Host
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    
    // Free Memory
    cudaFree(A_d); 
    cudaFree(B_d); 
    cudaFree(C_d);
}
```

---

## Q3.6: Single Element Data Index Mapping
**Question:** What is the expression for mapping thread and block indices to the data index `i` if each thread calculates exactly **one** output element of a vector addition?

**Answer:**
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

---

## Q3.7: Adjacent Elements Data Index Mapping
**Question:** Assume each thread calculates **two (adjacent)** elements of a vector addition. If `i` is the index for the first element to be processed by a thread, what is the mapping expression?

**Answer:**
```cpp
int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
```

---

## Q3.8: Calculating Grid Thread Count
**Question:** For a vector addition, assume the vector length is 2000, each thread calculates one output element, and the block size is 512 threads. How many total threads will be launched in the grid?

**Answer:**
**2048 threads.** 

Calculation:
```cpp
int threadNumber = ((2000 + 512 - 1) / 512) * 512;
// or
int threadNumber2 = ceil(2000 / 512.0) * 512; 
```
*(4 blocks × 512 threads per block = 2048 threads).*