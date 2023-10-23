#include "histogram.cuh"

///////////////////////// histogram 1 ///////////////////////////

__global__ void compute_histogram1(int* buffer, int* histogram, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size)
    {
        atomicAdd(&histogram[buffer[gid]], 1);
    }
}

///////////////////////// histogram 2 ///////////////////////////

__global__ void compute_histogram2(int* buffer, int* histogram, int bin, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid >= size) return;
    
    __shared__ int shared_histogram[bin];

    for (int i = threadIdx.x; i < bin; i += blockDim.x)
        shared_histogram[i] = 0;

    __syncthreads();

    if (gid < bin)
        atomicAdd(&shared_histogram[buffer[gid]], 1);

    __syncthreads();

    for (int i = threadIdx.x; i < bin; i += blockDim.x)
        atomicAdd(&histogram[i], shared_histogram[i]);
}