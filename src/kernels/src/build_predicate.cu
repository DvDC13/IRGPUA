#include "build_predicate.cuh"

__global__ void build_predicate(int* buffer, int* result, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        result[tid] = buffer[tid] != -27 ? 1 : 0;
    }
}

__global__ void build_predicate_zeros(int* buffer, int* result, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        result[tid] = buffer[tid] != 0 ? 1 : 0;
    }
}