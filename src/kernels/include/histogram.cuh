#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void compute_histogram(int* buffer, int* histogram, int size);