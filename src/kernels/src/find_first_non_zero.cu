#include "find_first_non_zero.cuh"

__global__ void find_first_non_zero(int* buffer, int* predicate, int *result, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        if (predicate[tid] == 1)
        {
            result[0] = buffer[tid];
        }
    }
}