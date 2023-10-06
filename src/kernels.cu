#include "kernels.cuh"

__global__ void reduce(int* buffer, int* total, int size)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= size) return;

    sdata[tid] = buffer[gid];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(total, sdata[0]);
    }
}