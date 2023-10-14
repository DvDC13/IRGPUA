#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void shift_buffer(int* buffer, int *result, int size);