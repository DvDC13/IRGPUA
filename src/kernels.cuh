#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void print_buffer(int *image_data, int size);

__global__ void build_predicate(int* buffer, int* result, int size);

__global__ void decoupled_lookback_scan(int* buffer, int* globalCounter, int* blocksA, int* blocksP, int* blockStates, int size);

__global__ void shift_buffer(int* buffer, int *result, int size);

__global__ void scatter_adresses(int* buffer, int* predicate, int size);

__global__ void apply_map(int* buffer, int size);

__global__ void compute_histogram(int* buffer, int* histogram, int size);

__global__ void create_predicate_zeros(int* buffer, int* result, int size);

__global__ void find_first_non_zero(int* buffer, int* predicate, int *result, int size);

__global__ void apply_map_transformation(int* buffer, int* histogram, int *first_non_zero, int size);

__global__ void kernel_reduce(int* buffer, int* total, int size);