#include "scan.cuh"

///////////////////////// scan 1 ///////////////////////////

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

///////////////////////// scan 2 ///////////////////////////

__global__ void decoupled_look_back_optimized(int *buffer, int size, int* d_blocks_aggregate, int* d_global_counter, cuda::std::atomic<char>* states)
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