#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void scatter_adresses(int* buffer, int* predicate, int size);