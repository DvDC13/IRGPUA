#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void kernel_reduce_baseline(const int* __restrict__ buffer, int* __restrict__ total, int size);

__global__ void reduce1(const int* __restrict__ buffer, int* __restrict__ total, int size);

__global__ void reduce2(const int* __restrict__ buffer, int* __restrict__ total, int size);

__global__ void reduce3(const int* __restrict__ buffer, int* __restrict__ total, int size);

__global__ void reduce4(const int* __restrict__ buffer, int* __restrict__ total, int size);

__global__ void reduce5(const int* __restrict__ buffer, int* __restrict__ total, int size);

__global__ void reduce6(const int* __restrict__ buffer, int* __restrict__ total, int blockSize, int size);

__global__ void reduce7(const int* __restrict__ buffer, int* __restrict__ total, int blockSize, int size);

__global__ void reduce8(const int* __restrict__ buffer, int* __restrict__ total, int size);

__global__ void reduce8(const int* __restrict__ buffer, int* __restrict__ total, int size);

__global__ void reduce9(const int* __restrict__ buffer, int* __restrict__ total, int size);