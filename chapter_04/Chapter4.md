# PMPP Chapter 4 — Data-Parallel Execution Model

## Exercise 4.1

**If a CUDA device’s SM can take up to 1,536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM?**
```text
a. 128 threads per block
b. 256 threads per block
c. 512 threads per block
d. 1,024 threads per block
```

```text
Answer:
a- 128*4 = 512 < 1536
b- 256*4 = 1024 < 1536
c- 512*3 = 1536 <= 1536
d- 1024*1 = 1024 < 1536
Hence, 512 threads per block with 3 blocks would result in exactly 1536 threads that SM can handle.
```


## Exercise 4.2
**For a vector addition, assume that the vector length is 2,000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?**
```text
a. 2,000
b. 2,024
c. 2,048
d. 2,096
```

```text
Answer:
512*3 = 1536 < 2000. Since each thread will calculate one output element, we need more threads which results in one extra block.
512*4 = 2048, 48 threads will not be used!
```

## Exercise 4.3
**For the previous question, how many warps do you expect to have divergence due to the boundary check on the vector length?**
```text
Answer:
a. 1
b. 2
c. 3
d. 6
```

```text
Answer:
Divergence only happens in a warp where some threads pass the boundary check and others fail it. A warp where all threads are out-of-bounds doesn't diverge — they all uniformly take the same branch (all skip), so there's no thread divergence there.

If the typical warp size is 32, then we would have 512/32=16 warps per block. The boundary (index 2000) falls in
the last block (threads 1536-2047, i.e. block 3). Locally, 2000 is at local index 464. Which warp is that? 464/32=14 -> warp 14 (global threads 1984-2015). Within that warp,
local offset 464-14*32=16, so threads 0-15 of that warp are valid (1984-1999) and threads 16-31 are invalid (2000-2015), this warp diverges.
The next warp, warp 15 (global 2016-2047), is entirely out of bounds, uniform, no divergence. Hence, only 1 warp diverges.
```

## Exercise 4.4
**You need to write a kernel that operates on an image of size 400 x 900 pixels. You would like to assign one thread to each pixel. You would like your thread blocks to be square and to use the maximum number of threads per block possible on the device (your device has compute capability 3.0). How would you select the grid dimensions and block dimensions of your kernel?**

```text
Answer:
Due to compute capability 3.0, maximum number of threads that a block can have is hardware limited to 1024.
Since we would like our thread blocks to be square and use the maximum number of threads per block, we will select 32x32 as our block dimension.
Thus, gridDim.x = ceil(900/32)=29 and gridDim.y = ceil(400/32)=13.
So,
dim3 blockDim(32, 32, 1); dim3 gridDim(29, 13, 1); (mapping width=900→x, height=400→y), which gives 29×32=928 ≥ 900 and 13×32=416 ≥ 400 — covers the whole image with one thread per pixel plus boundary-checked overhang.
```

## Exercise 4.5
**For the previous question, how many idle threads do you expect to have?**

```text
Answer:
Total launched threads: (29*32)*(13*32)=386,048
Actual pixels needing work: 400*900=360,000
Idle threads: 386,048 - 360,000 = 26,048
928-900=28 extra columns of threads across the full 416 height, plus 416-400=16 extra rows across the full 928 width, with the small 28x16 corner double-counted.
In CUDA, by convention:
x → horizontal direction → columns → maps to width
y → vertical direction → rows → maps to height

This is the opposite of C array notation, where the first index (i in arr[i][j]) is the row. In CUDA, x is not "the first dimension" in that row/col sense — it's the horizontal axis, full stop.
So when computing a pixel's coordinates inside the kernel, you'd typically write:
```
```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;  // x → width direction
int row = blockIdx.y * blockDim.y + threadIdx.y;  // y → height direction
```
```text
Given the image is 400 (height) × 900 (width):

Width = 900 → needs ceil(900/32) = 29 blocks → this is gridDim.x
Height = 400 → needs ceil(400/32) = 13 blocks → this is gridDim.y

So gridDim(29, 13, 1) is correct — but the reason is "x tracks width" and "y tracks height," not "x is the first/row dimension."
The confusion is easy to fall into because textbooks often write image dimensions as height × width (matching how you'd index image[row][col] in memory), while CUDA's launch config is written (x, y, z) = (width-direction, height-direction, depth). Same numbers, different ordering convention — worth double-checking every time you set up a 2D kernel until it's automatic.
```

## Exercise 4.6
**Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9, and spend the rest of their time waiting for the barrier. What percentage of the threads’ summed-up execution times is spent waiting for the barrier?**
```text
Answer:
At most 3 seconds is required to complete the section before reaching the barrier (3'rd thread). Hence, 3 seconds will be required for each thread to enter the barrier.
(2 + 2.3 + 3 + 2.8 + 2.4 + 1.9 + 2.6 + 2.9) / 24 * 100 = 82.917% time computing
100-82.917=17.083%  time waiting
```

## Exercise 4.7
**Indicate which of the following assignments per multiprocessor is possible. In the case where it is not possible, indicate the limiting factor(s).**
```text
a. 8 blocks with 128 threads each on a device with compute capability 1.0
b. 8 blocks with 128 threads each on a device with compute capability 1.2
c. 8 blocks with 128 threads each on a device with compute capability 3.0
d. 16 blocks with 64 threads each on a device with compute capability 1.0
e. 16 blocks with 64 threads each on a device with compute capability 1.2
f. 16 blocks with 64 threads each on a device with compute capability 3.0
```
| Configuration | Total Threads | Compute Capability | Max Threads / SM | Max Blocks / SM | Can all blocks run simultaneously? | Limiting Factor / Status |
| :------------ | ------------: | :----------------- | ---------------: | --------------: | :--------------------------------: | :----------------------- |
| **a. 8 × 128 threads** | 1024 | **1.0 (Tesla)** | 768 | 8 | **No** | Exceeds maximum threads per SM (1024 > 768) |
| **b. 8 × 128 threads** | 1024 | **1.2 (Tesla)** | 1024 | 8 | **Yes** | Fully capable (reaches both thread and block limits exactly) |
| **c. 8 × 128 threads** | 1024 | **3.0 (Kepler)** | 2048 | 16 | **Yes** | Fully capable (50% thread utilization, 50% block utilization) |
| **d. 16 × 64 threads** | 1024 | **1.0 (Tesla)** | 768 | 8 | **No** | Exceeds both thread limit (1024 > 768) and block limit (16 > 8) |
| **e. 16 × 64 threads** | 1024 | **1.2 (Tesla)** | 1024 | 8 | **No** | Exceeds maximum blocks per SM (16 > 8) |
| **f. 16 × 64 threads** | 1024 | **3.0 (Kepler)** | 2048 | 16 | **Yes** | Fully capable (50% thread utilization, reaches block limit exactly) |

## Exercise 4.8
**A CUDA programmer says that if they launch a kernel with only 32 threads in each block,they can leave out the __syncthreads() instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain.**
```text
Answer:
A block of exactly 32 threads — exactly one warp. The programmer's reasoning is presumably "since 32 threads all fit in a single warp, and warp threads execute in lockstep (SIMD), they're automatically synchronized already, so __syncthreads() is redundant."
Why this is not a good idea:

It relies on an implementation detail, not a guarantee of the programming model. The CUDA programming model does not promise that all threads in a warp execute in lockstep — that's historically been true as a hardware implementation detail (SIMT execution), but it's not part of the language's formal semantics. This assumption is already broken on modern hardware. Starting with Volta (compute capability 7.0), NVIDIA introduced independent thread scheduling, where threads within a warp can diverge and make independent progress — they are no longer guaranteed to stay in lockstep even without explicit branches. Code that skips __syncthreads() based on warp-implicit-sync assumptions can silently produce race conditions or stale/inconsistent shared memory reads on these architectures. Portability/future-proofing. Even if it "worked" on some older GPU, the code becomes fragile and non-portable across CUDA versions and hardware generations — a change the programmer has no control over.
```

## Exercise 4.9
**A student mentioned that he was able to multiply two 1,024x1,024 matrices using a tiled matrix multiplication code with 32x32 thread blocks. He is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. He further mentioned that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?**
```text
First, the student's original claim needs correcting on two counts:

A 32×32 block is 1,024 threads — this exceeds the device's 512-threads-per-block limit as originally stated, so that configuration would fail to launch at all. (Even granting a hypothetical 1,024-thread-per-block limit, it would still be a poor choice for occupancy, as shown below.)
Each thread computing one output element is not itself a problem — it just means the grid must contain enough threads in total to cover all 1024×1024 = 1,048,576 output elements. This is independent of how many threads can be resident on an SM at once.

Assuming instead a 16×16 thread block (256 threads/block), which fits comfortably under a 512-thread-per-block limit:
Grid size: Since one thread computes one output element and the result matrix is 1024×1024:

gridDim.x = ceil(1024 / 16) = 64
gridDim.y = ceil(1024 / 16) = 64
Total blocks in the grid = 64 × 64 = 4,096 blocks, 256 threads each → 1,048,576 threads total, matching the number of output elements exactly (no idle threads/boundary padding needed here, since 1024 is evenly divisible by 16).

SM occupancy check (using the device's per-SM limits — 1,536 max resident threads, up to 8 blocks per SM in the exercise's original numbers):

Thread limit: 1,536 / 256 = 6 blocks' worth of threads fit before hitting the thread cap.
Block-count limit: up to 8 blocks allowed per SM — not binding, since we hit the thread cap first at 6.
So 6 blocks (1,536 threads) reside per SM simultaneously — full 100% thread occupancy, with 2 of the 8 available block slots left unused (not a problem, since there's no more thread capacity to fill them with anyway).

Conclusion: With 16×16 blocks, the SM's thread capacity is fully utilized (1,536/1,536), which is a much better occupancy outcome than the 1,024/1,536 (≈67%) you'd get from a 32×32 block. The tradeoff is that smaller blocks mean more blocks total need to be scheduled across the grid (4,096 vs. far fewer with 32×32), and the tiled-matmul kernel's shared-memory tile size must match the smaller block dimensions (16×16 tiles instead of 32×32), which typically means more loop iterations over tiles to cover the same K-dimension — a standard tradeoff between occupancy and per-block work efficiency.

Tradeoff 1: Occupancy (latency hiding within an SM)

16×16 blocks → 6 blocks resident per SM → 1,536/1,536 threads = 100% occupancy
32×32 blocks → 1 block resident per SM → 1,024/1,536 threads = 67% occupancy

More resident warps means the SM has more independent work to switch to when one warp stalls (e.g., waiting on a global memory load). Higher occupancy generally helps hide that latency.
Tradeoff 2: Data reuse from shared memory (global memory traffic)
This is the one the conclusion glossed over, and it cuts the other way. In tiled matmul, each tile of A and B loaded into shared memory gets reused by every thread in the block before being discarded:

With a 32×32 tile, each element loaded from global memory into shared memory is reused by 32 threads (once per row/column of the tile).
With a 16×16 tile, each loaded element is only reused by 16 threads.

So the 32×32 configuration needs half as many passes over the K-dimension (32 iterations vs. 64, for a K=1024 matrix) and issues fewer total global memory transactions per output element — better arithmetic intensity, less global memory bandwidth pressure.
So which is "more logical"?
There's no universally correct answer — it depends on which resource is your actual bottleneck:

If the kernel is latency-bound (SM often idle waiting on memory, and there isn't other work to hide it) → the 16×16 config's full occupancy wins, since it keeps more warps in flight to cover those stalls.
If the kernel is memory-bandwidth-bound (the GPU's DRAM bandwidth is the ceiling, not latency) → the 32×32 config's better data reuse wins, since it reduces the total bytes moved from global memory, which is often the actual limiter for compute-dense kernels like matmul.

In practice, dense matrix multiplication tends to be memory-bandwidth-bound at small tile sizes, which is why larger tiles are usually preferred in real tuned kernels — up to the point where shared memory or register pressure caps how large you can go. That's also why production GEMM kernels typically don't stop at "one thread per output element" with a single tile size — they use multi-level tiling (each thread computes several output elements) to get both large effective tile reuse and enough independent threads for occupancy simultaneously.
The honest conclusion for this exercise: it's a genuine tradeoff between occupancy and memory-traffic efficiency, and picking a winner requires profiling on the actual hardware rather than reasoning from occupancy numbers alone.
```

## Exercise 4.10
**The following kernel is executed on a large matrix, which is tiled into submatrices. To manipulate tiles, a new CUDA programmer has written the following device kernel, which will transpose each tile in the matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of matrix A is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. BLOCK_WIDTH is known at compile time, but could be set anywhere from 1 to 20.**

```cpp
dim3 blockDim(BLOCK_WIDTH,BLOCK_WIDTH);
dim3 gridDim(A_width/blockDim.x,A_height/blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

__global__ void
BlockTranspose(float* A_elements, int A_width, int A_height)
{
__shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];
int baseIdx = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
baseIdx += (blockIdx.y * BLOCK_WIDTH + threadIdx.y) * A_width;
blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```
```text
a. Out of the possible range of values for BLOCK_WIDTH, for what values of BLOCK_WIDTH will this kernel function correctly when executing on the device?

b. If the code does not execute correctly for all BLOCK_WIDTH values, suggest a fix to the code to make it work for all BLOCK_WIDTH values.
```

```cpp
blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];   // write to shared memory
A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];   // read a *different* location
```
```text
Each thread reads a shared-memory location that was written by a different thread (the transposed position). Without a barrier, there's no guarantee that the thread responsible for writing blockA[threadIdx.x][threadIdx.y] has actually done so before the reading thread executes its line — a classic shared-memory race condition.

a. For which BLOCK_WIDTH values does it happen to work anyway?
It "works" only when the entire thread block fits inside a single warp (32 threads), because — historically, and setting aside the independent-thread-scheduling caveat from Volta+ we discussed earlier — threads within one warp execute in lockstep, so the write instruction completes for the whole warp before any thread in that same warp moves on to the read instruction. No separate warps are involved, so there's no cross-warp race.
A block is BLOCK_WIDTH × BLOCK_WIDTH threads. That must be ≤ 32:

BLOCK_WIDTH = 5 → 25 threads ✓ (fits in one warp)
BLOCK_WIDTH = 6 → 36 threads ✗ (spans 2 warps)

So BLOCK_WIDTH = 1 through 5 happen to execute correctly (relying on same-warp lockstep execution), and BLOCK_WIDTH = 6 through 20 will produce race conditions and incorrect results, since the block spans multiple warps that aren't guaranteed to execute in any particular relative order.
Worth flagging, tying back to the earlier question: even the "working" range (1–5) is only correct because of an implementation detail (warp lockstep), not because the code is actually correct per the CUDA programming model. It's fragile in the same way — technically undefined behavior that happens to pass today.
b. The fix:
```
```cpp
__global__ void
BlockTranspose(float* A_elements, int A_width, int A_height)
{
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_WIDTH + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

    __syncthreads();   // ensure all threads finish writing before any thread reads

    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```
```text
This guarantees every thread in the block has finished writing its element into shared memory before any thread proceeds to read the transposed element — making the kernel correct for all BLOCK_WIDTH values from 1 to 20, not just the ones that happen to fit in a single warp.

Why gridDim is irrelevant here:
gridDim (computed as A_width/BLOCK_WIDTH and A_height/BLOCK_WIDTH) just determines how many blocks are launched to cover the whole matrix. Each block operates completely independently — it has its own private shared-memory array block, and one block's execution has zero interaction with another block's. The number of blocks in the grid has no bearing on whether the race condition inside a single block occurs. You could have a grid of 2 blocks or 2 million blocks — it wouldn't change whether the bug appears, because the bug lives entirely within one block's execution.
What actually determines correctness:
It's the block dimensions (blockDim = BLOCK_WIDTH × BLOCK_WIDTH threads, not blocks) relative to the warp size (32 threads) — a completely different quantity than grid size. The race only fails to manifest when an entire block's threads fit inside a single warp, so they execute in lockstep with no possibility of the read outrunning the write:

BLOCK_WIDTH × BLOCK_WIDTH ≤ 32 → whole block is one warp → no cross-warp race → happens to work (BLOCK_WIDTH = 1 to 5)
BLOCK_WIDTH × BLOCK_WIDTH > 32 → block spans multiple warps → race condition → BLOCK_WIDTH = 6 to 20 broken

So the corrected statement should reference BLOCK_WIDTH² vs. warp size (32), not gridDim. The "outside of the BLOCK_WIDTH value" phrasing in your version also doesn't quite match the actual failure mode — the bug isn't about elements outside some range not getting transposed; it's that the entire tile's transpose becomes unreliable/incorrect wherever cross-warp read-after-write ordering isn't guaranteed, because different threads may read stale or not-yet-written shared-memory values.


dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
This explicitly sets blockDim.x = BLOCK_WIDTH and blockDim.y = BLOCK_WIDTH. So inside the kernel, blockDim.x, blockDim.y, and the compile-time constant BLOCK_WIDTH all hold the exact same value — using one versus the other in the index math produces identical results

In fact, using the compile-time constant BLOCK_WIDTH instead of the runtime blockDim.x/blockDim.y is a common and often preferred pattern in tiled CUDA kernels — since BLOCK_WIDTH is known at compile time, the compiler can potentially do more aggressive optimization (constant folding, loop unrolling, register allocation) than it could with a runtime-read blockDim.x. It's a style/performance choice, not a correctness issue.
So to be clear about where the actual bug is: it's purely the missing __syncthreads() between the shared-memory write and the transposed read. The indexing logic and the use of BLOCK_WIDTH (once corrected from the BLOCK_SIZE typo) are both correct as written.

```