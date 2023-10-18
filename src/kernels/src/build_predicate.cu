#include "build_predicate.cuh"

///////////////////////// build_predicate 1 ///////////////////////////

__global__ void build_predicate1(const int* __restrict__ buffer, int* __restrict__ result, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size)
    {
        result[gid] = buffer[gid] != -27 ? 1 : 0;
    }
}

///////////////////////// build_predicate 2 ///////////////////////////

__global__ void build_predicate2(const int* __restrict__ buffer, int* __restrict__ result, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size)
    {
        constexpr int garbage_value = -27;
        result[gid] = buffer[gid] != garbage_value;
    }
}

///////////////////////// build_predicate_zeros 1 ///////////////////////////

__global__ void build_predicate_zeros1(const int* __restrict__ buffer, int* __restrict__ result, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size)
    {
        result[gid] = buffer[gid] != 0;
    }
}