#include "find_first_non_zero.cuh"

__global__ void find_first_non_zero(int* buffer, int* predicate, int *result, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size)
    {
        if (predicate[gid] == 1)
        {
            result[0] = buffer[gid];
        }
    }
}