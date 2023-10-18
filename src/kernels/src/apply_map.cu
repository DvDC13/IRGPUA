#include "apply_map.cuh"

///////////////////////// apply_map 1 ///////////////////////////

__global__ void apply_map1(int* buffer, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size)
    {
        if (gid % 4 == 0)
        {
            buffer[gid] += 1;
        }
        else if (gid % 4 == 1)
        {
            buffer[gid] -= 5;
        }
        else if (gid % 4 == 2)
        {
            buffer[gid] += 3;
        }
        else if (gid % 4 == 3)
        {
            buffer[gid] -= 8;
        }
    }
}

///////////////////////// apply_map 2 ///////////////////////////

__global__ void apply_map2(int* buffer, int size, int* map)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size)
    {
        buffer[gid] += map[gid % 4];
    }
}

///////////////////////// apply_map 3 ///////////////////////////

__device__ __constant__ int map_gpu[4] = {1, -5, 3, -8};

__global__ void apply_map3(int* buffer, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size)
    {
        buffer[gid] += map_gpu[gid % 4];
    }
}

///////////////////////// apply_map 4 ///////////////////////////

__global__ void apply_map4(int* buffer, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size)
    {
        buffer[gid] += map_gpu[gid & 3];
    }
}