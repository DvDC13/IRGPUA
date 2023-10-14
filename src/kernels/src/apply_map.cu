#include "apply_map.cuh"

__global__ void apply_map(int* buffer, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        if (tid % 4 == 0)
        {
            buffer[tid] += 1;
        }
        else if (tid % 4 == 1)
        {
            buffer[tid] -= 5;
        }
        else if (tid % 4 == 2)
        {
            buffer[tid] += 3;
        }
        else if (tid % 4 == 3)
        {
            buffer[tid] -= 8;
        }
    }
}