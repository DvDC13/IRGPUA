#include "apply_map_transformation.cuh"

__global__ void apply_map_transformation(int* buffer, int* histogram, int *first_non_zero, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        buffer[tid] = std::roundf(((histogram[buffer[tid]] - first_non_zero[0]) / (float)(size - first_non_zero[0])) * 255.0f);
    }
}