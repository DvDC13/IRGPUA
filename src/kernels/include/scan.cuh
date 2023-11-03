#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda/atomic>

__global__ void decoupled_look_back(int *buffer, int* d_blocks_aggregate, int* d_global_counter, cuda::std::atomic<char>* states, int size);

__global__ void scan_look_back_cascade(int *buffer, int size, int* d_global_counter, cuda::std::atomic<char>* states);