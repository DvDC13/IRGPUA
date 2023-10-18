#include "scatter_adresses.cuh"

__global__ void scatter_adresses(int* buffer, int* predicate, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size)
    {
        if (buffer[gid] != -27)
        {
            buffer[predicate[gid]] = buffer[gid];
        }
    }
}