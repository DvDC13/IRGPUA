#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void compute_histogram1(int* buffer, int* histogram, int size);

__global__ void compute_histogram2(int* buffer, int* histogram, int size);