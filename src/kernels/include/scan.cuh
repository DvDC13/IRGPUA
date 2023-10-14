#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void decoupled_lookback_scan(int* buffer, int* globalCounter, int* blocksA, int* blocksP, int* blockStates, int size);