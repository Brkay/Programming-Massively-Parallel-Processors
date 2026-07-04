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

# 5. Link the chapter modules
add_subdirectory(chapter_03)

# Future chapters will simply be added here:
# add_subdirectory(chapter_04)
# add_subdirectory(chapter_05)