#include "scan.cuh"

///////////////////////// scan 1 ///////////////////////////

__global__ void decoupled_look_back(int *buffer, int* d_blocks_aggregate, int* d_global_counter, cuda::std::atomic<char>* states, int size)
{
    __shared__ int blockIndex;
    int tid = threadIdx.x;
    if (tid == 0)
        blockIndex = atomicAdd(d_global_counter, 1);
    __syncthreads();

    int gid = blockIndex * blockDim.x + tid;

    if (gid >= size) return;

    int idx_offset = gid - blockIndex * blockDim.x;

    for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1)
    {
        int index = idx_offset - offset;
        int val;
        if (index >= 0)
            val = buffer[gid - offset];
        __syncthreads();
        if (index >= 0) {
            buffer[gid] += val;
        }
        __syncthreads();
    }

    if (blockIndex == 0)
    {
        d_blocks_aggregate[0] = buffer[blockIndex * blockDim.x + blockDim.x - 1];
        states[blockIndex].store('P', cuda::std::memory_order_release);
    }
    else
    {
        d_blocks_aggregate[blockIndex] = buffer[blockIndex * blockDim.x + blockDim.x - 1];
        states[blockIndex].store('A', cuda::std::memory_order_release);
    }

    __syncthreads();

    if (blockIndex > 0)
    {
        int sum = 0;
        int i_block = blockIndex - 1;
        char state = states[i_block].load(cuda::std::memory_order_acquire);

        while (state != 'P')
        {
            if (state == 'A')
            {
                sum += d_blocks_aggregate[i_block];
                i_block--;
            }

            state = states[i_block].load(cuda::std::memory_order_acquire);
        }

        sum += buffer[i_block * blockDim.x + blockDim.x - 1];
        atomicAdd(&buffer[gid], sum);

        __syncthreads();

        states[blockIndex].store('P', cuda::std::memory_order_release);
    }
}

///////////////////////// scan 3 ///////////////////////////

__global__ void scan_look_back_cascade(int *buffer, int size, int* d_global_counter, cuda::std::atomic<char>* states)
{
    __shared__ int blockIndex;
    int tid = threadIdx.x;
    if (tid == 0)
        blockIndex = atomicAdd(d_global_counter, 1);
    __syncthreads();

    int gid = blockIndex * blockDim.x + tid;

    if (gid >= size) return;

    int idx_offset = gid - blockIndex * blockDim.x;

    for (unsigned int offset = 1; offset <= blockDim.x / 2; offset *= 2)
    {
        int val;
        if (idx_offset >= offset)
            val = buffer[gid - offset];
        __syncthreads();
        if (idx_offset >= offset) {
            buffer[gid] += val;
        }
        __syncthreads();
    }

    if (blockIndex == 0)
        states[blockIndex].store('P', cuda::std::memory_order_release);
    else
        states[blockIndex].store('A', cuda::std::memory_order_release);

    __syncthreads();

    if (blockIndex > 0)
    {
        int value = 0;
        int i_block = blockIndex - 1;
        char state = states[i_block].load(cuda::std::memory_order_acquire);

        while (state != 'P')
        {
            state = states[i_block].load(cuda::std::memory_order_acquire);
        }

        value = buffer[i_block * blockDim.x + blockDim.x - 1];
        atomicAdd(&buffer[gid], value);
        states[blockIndex].store('P', cuda::std::memory_order_release);
    }
}