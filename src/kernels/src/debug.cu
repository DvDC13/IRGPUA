#include "debug.cuh"

__global__ void print_buffer(int *buffer, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid == 0)
    {
        for (int i = 0; i < size; i++)
            printf("image[%d] = %d\n", i, buffer[i]);
    }
    __syncthreads();
}