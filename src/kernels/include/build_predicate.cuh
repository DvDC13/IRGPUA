#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void build_predicate1(const int* __restrict__ buffer, int* __restrict__ result, int size);

__global__ void build_predicate2(const int* __restrict__ buffer, int* __restrict__ result, int size);

__global__ void build_predicate3(const int* __restrict__ buffer, int* __restrict__ result, int size);

__global__ void build_predicate_zeros1(const int* __restrict__ buffer, int* __restrict__ result, int size);