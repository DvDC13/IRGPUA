#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void print_buffer(int *image_data, int size);