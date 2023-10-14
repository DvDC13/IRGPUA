#include "histogram.cuh"

__global__ void compute_histogram(int* buffer, int* histogram, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        atomicAdd(&histogram[buffer[tid]], 1);
    }
}