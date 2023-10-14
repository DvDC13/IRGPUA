#include "scatter_adresses.cuh"

__global__ void scatter_adresses(int* buffer, int* predicate, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        if (buffer[tid] != -27)
        {
            buffer[predicate[tid]] = buffer[tid];
        }
    }
}