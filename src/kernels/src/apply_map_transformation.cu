#include "apply_map_transformation.cuh"

///////////////////////// apply_map_transformation 1 ///////////////////////////

__global__ void apply_map_transformation1(int* buffer, int* histogram, int *first_non_zero, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size)
    {
        buffer[gid] = std::roundf(((histogram[buffer[gid]] - first_non_zero[0]) / (float)(size - first_non_zero[0])) * 255.0f);
    }
}