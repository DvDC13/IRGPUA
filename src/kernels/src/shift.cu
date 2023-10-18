#include "shift.cuh"

__global__ void shift_buffer(int* buffer, int *result, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size)
    {
        result[gid] = gid == 0 ? 0 : buffer[gid - 1];
    }
}