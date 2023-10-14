#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void apply_map(int* buffer, int size);