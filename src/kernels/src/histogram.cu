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

__global__ void compute_histogram2(int* buffer, int* histogram, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size)
    {
        __shared__ int shared_histogram[256];

        if (threadIdx.x < 256)
        {
            shared_histogram[threadIdx.x] = 0;
        }

        __syncthreads();

        atomicAdd(&shared_histogram[buffer[gid]], 1);

        __syncthreads();

        atomicAdd(&histogram[threadIdx.x], shared_histogram[threadIdx.x]);
    }
}