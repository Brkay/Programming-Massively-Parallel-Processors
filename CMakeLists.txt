cmake_minimum_required(VERSION 3.20)

# 1. Point CMake to the local CUDA compiler on your Ubuntu laptop
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# 2. Define the project and enable both C++ and CUDA
project(pmpp_exercises LANGUAGES CXX CUDA)

# 3. Enforce modern language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 4. Target the RTX 4080 (Ada Lovelace / Compute Capability 8.9) on your desktop
set(CMAKE_CUDA_ARCHITECTURES 89)

# ==========================================
# RELEASE PROFILE FLAGS (Aggressive Speed)
# ==========================================

# Host (CPU) Release Flags for GCC:
# -O3: Maximize CPU optimization
# -march=native: Optimize for your exact CPU architecture
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Device (GPU) Release Flags for NVCC:
# -O3: Maximize GPU optimization
# -use_fast_math: Sacrifices slight floating-point precision for massive speed gains
# --ptxas-options=-v: Prints out register and shared memory usage during compilation (very useful!)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math --ptxas-options=-v")


# ==========================================
# DEBUG PROFILE FLAGS (Strict Checking)
# ==========================================

# Host (CPU) Debug Flags:
# -O0: Disable optimizations so code steps linearly in the debugger
# -g: Generate CPU debug symbols
# -Wall -Wextra: Turn on all C++ warnings
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")

# Device (GPU) Debug Flags:
# -O0: Disable GPU optimizations
# -g: Generate host debug symbols
# -G: Generate device debug symbols (Required to step inside a kernel using cuda-gdb)
# -lineinfo: Maps compiled instructions back to your source code lines
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G -lineinfo")

# 5. Link the chapter modules
add_subdirectory(chapter_03)

# Future chapters will simply be added here:
# add_subdirectory(chapter_04)
# add_subdirectory(chapter_05)