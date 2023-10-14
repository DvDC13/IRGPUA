#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void build_predicate(int* buffer, int* result, int size);

__global__ void build_predicate_zeros(int* buffer, int* result, int size);