#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void apply_map_transformation(int* buffer, int* histogram, int *first_non_zero, int size);