#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda/atomic>

__global__ void decoupled_lookback_scan(int* buffer, int* globalCounter, int* blocksA, int* blocksP, int* blockStates, int size);

__global__ void decoupled_look_back_optimized(int *buffer, int size, int* d_blocks_aggregate, int* d_global_counter, cuda::std::atomic<char>* states);

__global__ void scan_look_back_cascade(int *buffer, int size, int* d_global_counter, cuda::std::atomic<char>* states);