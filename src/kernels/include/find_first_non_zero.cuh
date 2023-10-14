#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void find_first_non_zero(int* buffer, int* predicate, int *result, int size);