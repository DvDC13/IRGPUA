#include "shift.cuh"

__global__ void shift_buffer(int* buffer, int *result, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        result[tid] = tid == 0 ? 0 : buffer[tid - 1];
    }
}