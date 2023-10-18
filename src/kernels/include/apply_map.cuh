#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void apply_map1(int* buffer, int size);

__global__ void apply_map2(int* buffer, int size);

__global__ void apply_map3(int* buffer, int size);

__global__ void apply_map4(int* buffer, int size);