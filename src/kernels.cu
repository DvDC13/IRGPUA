#include "kernels.cuh"

__global__ void print_buffer(int *buffer, int size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid == 0)
    {
        for (int i = 0; i < size; i++)
            printf("image[%d] = %d\n", i, buffer[i]);
    }
    __syncthreads();
}

__global__ void build_predicate(int* buffer, int* result, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        result[tid] = buffer[tid] != -27 ? 1 : 0;
    }
}

__global__ void decoupled_lookback_scan(int* buffer, int* globalCounter, int* blocksA, int* blocksP, int* blockStates, int size)
{
    // Block ordering
    __shared__ int blockIndex;
    int tid = threadIdx.x;
    if (tid == 0)
        blockIndex = atomicAdd(globalCounter, 1);
    __syncthreads();

    // Define the global index
    int gid = blockIndex * blockDim.x + tid;

    // Check if the global index is valid
    if (gid >= size) return;

    int idx_offset = gid - blockIndex * blockDim.x;
    // Local scan of the block
    for (unsigned int offset = 1; offset <= blockDim.x / 2; offset *= 2)
    {
        int val;
        if (idx_offset >= offset)
            val = buffer[gid - offset];
        __syncthreads();
        if (idx_offset >= offset)
            buffer[gid] += val;
        __syncthreads();
    }

    // Add the last element of the block to blocksP or blocksA and set the block state to 1 or 2
    if (tid == blockDim.x - 1)
    {
        if (blockIndex == 0)
            atomicAdd(&blocksP[blockIndex], buffer[gid]);
        else
            atomicAdd(&blocksA[blockIndex], buffer[gid]);

        __threadfence();

        blockStates[blockIndex] = 1 + (blockIndex == 0);
    }

    // Make sure that each thread reads the correct value of blockStates
    __syncthreads();

    // Run decoupled lookback scan
    if (blockIndex > 0)
    {
        if (tid == 0)
        {
            int i_block = blockIndex - 1;
            int state = atomicAdd(&blockStates[i_block], 0);

            while (state != 2)
            {
                if (state == 1)
                {
                    int previous_block_value = atomicAdd(&blocksA[i_block], 0);
                    blocksP[blockIndex] += previous_block_value;
                    i_block--;
                }

                state = atomicAdd(&blockStates[i_block], 0);
            }

            __threadfence();

            // Found a block with state 2 (prefix sum available)
            blocksP[blockIndex] += atomicAdd(&blocksP[i_block], 0) + atomicAdd(&blocksA[i_block], 0);
            blockStates[blockIndex] = 2;
        }

        __syncthreads();

        buffer[gid] += atomicAdd(&blocksP[blockIndex], 0);
    }
}

__global__ void shift_buffer(int* buffer, int *result, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        result[tid] = tid == 0 ? 0 : buffer[tid - 1];
    }
}

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

__global__ void apply_map(int* buffer, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        if (tid % 4 == 0)
        {
            buffer[tid] += 1;
        }
        else if (tid % 4 == 1)
        {
            buffer[tid] -= 5;
        }
        else if (tid % 4 == 2)
        {
            buffer[tid] += 3;
        }
        else if (tid % 4 == 3)
        {
            buffer[tid] -= 8;
        }
    }
}

__global__ void compute_histogram(int* buffer, int* histogram, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        atomicAdd(&histogram[buffer[tid]], 1);
    }
}

__global__ void create_predicate_zeros(int* buffer, int* result, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        result[tid] = buffer[tid] != 0 ? 1 : 0;
    }
}

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

__global__ void apply_map_transformation(int* buffer, int* histogram, int *first_non_zero, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        buffer[tid] = std::roundf(((histogram[buffer[tid]] - first_non_zero[0]) / (float)(size - first_non_zero[0])) * 255.0f);
    }
}

__global__ void kernel_reduce(int* buffer, int* total, int size)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = buffer[gid];

    if (gid >= size) return;

    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(total, sdata[0]);
    }
}