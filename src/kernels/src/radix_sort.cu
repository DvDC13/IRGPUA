#include "radix_sort.cuh"

__global__ void order_checking(int* d_reduce, int* d_total, int size)
{
    extern __shared__ int s_arr[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < size)
    {
        s_arr[tid] = d_reduce[gid];
    }
    else
    {
        s_arr[tid] = 0;
    }

    __syncthreads();

    // Perform order checking
    if (gid < (size - 1))
    {
        s_arr[gid] = s_arr[gid] > s_arr[gid + 1];
    }
    s_arr[size - 1] = 0;
    __syncthreads();

    // Perform optimized reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_arr[tid] += s_arr[tid + offset];
        }
        __syncthreads();
    }

    // Write the block sum to global memory
    if (tid == 0)
    {
        atomicAdd(d_total, s_arr[0]);
    }
}

__global__ void radix_sort(int* d_arr_in, int* d_blocks_sum, int* d_prefix_sum, int bit_shift, int* d_arr_out, int size)
{
    extern __shared__ int s_arr[];

    int* s_data = s_arr;
    int* s_mask = s_arr + blockDim.x;
    int* s_local_prefix_sum = s_arr + blockDim.x + blockDim.x + 1;
    int* s_mask_scan = s_arr + blockDim.x + blockDim.x + 1 + blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < size)
    {
        s_data[tid] = d_arr_in[gid];
    }
    else
    {
        s_data[tid] = 0;
    }

    __syncthreads();

    unsigned int data = s_data[tid];
    unsigned int bits = (data >> bit_shift) & 3;

    for (int b = 0; b <= 3; b++)
    {
        // Create the mask
        s_mask[tid] = 0;
        __syncthreads();
        bool is_equal = false;
        if (gid < size)
        {
            is_equal = (bits == b);
            s_mask[tid] = is_equal;
        }
        __syncthreads();

        // Perform scan on the mask
        int tmp = 0;
        for (int d = 0; d < (int)log2f(blockDim.x); d++)
        {
            int before = tid - (1 << d);

            if (before >= 0)
                tmp = s_mask[before] + s_mask[tid];
            else
                tmp = s_mask[tid];
            __syncthreads();
            s_mask[tid] = tmp;
            __syncthreads();
        }

        __syncthreads();

        // Shift to the right to produce an exclusive prefix sum
        s_mask[tid + 1] = s_mask[tid];

        __syncthreads();

        if (tid == 0)
        {
            s_mask[0] = 0;
            int total = s_mask[blockDim.x];
            int block_index = gridDim.x * b + blockIdx.x;
            d_blocks_sum[block_index] = total;

            s_mask_scan[b] = total;
        }

        __syncthreads();

        if (is_equal && gid < size)
        {
            s_local_prefix_sum[tid] = s_mask[tid];
        }
    }

    // Perform scan on the d_mask_scan
    int tmp = 0;
    for (int d = 0; d < (int)log2f(blockDim.x); d++)
    {
        int before = tid - (1 << d);

        if (before >= 0)
            tmp = s_mask_scan[before] + s_mask_scan[tid];
        else
            tmp = s_mask_scan[tid];
        __syncthreads();
        s_mask_scan[tid] = tmp;
        __syncthreads();
    }

    __syncthreads();

    // Shift to the right to produce an exclusive prefix sum
    s_mask_scan[tid + 1] = s_mask_scan[tid];

    if (tid == 0)
    {
        s_mask_scan[0] = 0;
    }

    __syncthreads();

    if (gid < size)
    {
        int t = s_local_prefix_sum[tid];
        int new_position = t + s_mask_scan[bits];
        
        __syncthreads();

        s_data[new_position] = data;
        s_local_prefix_sum[new_position] = t;

        __syncthreads();

        d_prefix_sum[gid] = s_local_prefix_sum[tid];
        d_arr_out[gid] = s_data[tid];
    }
}

__global__ void compute_new_position(int *d_arr_out, int* d_arr_in, int* d_prefix_sum, int* d_scan_blocks_sum, int bit_shift, int size)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid >= size) return;

    int data = d_arr_out[gid];
    int bits = (data >> bit_shift) & 3;
    int m = d_prefix_sum[gid];
    int final_index = bits * gridDim.x + blockIdx.x;
    int final_position = d_scan_blocks_sum[final_index] + m;
    __syncthreads();
    d_arr_in[final_position] = data;
}

void radix_sort_gpu(int* arr, int size)
{
    int block_size = 4;
    int grid_size = (size + block_size - 1) / block_size;

    int shared_memory_size = sizeof(int) * (
        block_size + (block_size + 1) + block_size + 8
    );

    int* d_arr_in;
    cudaMalloc((void**)&d_arr_in, size * sizeof(int));
    cudaMemset(d_arr_in, 0, size * sizeof(int));
    cudaMemcpy(d_arr_in, arr, size * sizeof(int), cudaMemcpyHostToDevice);

    int* d_arr_out;
    cudaMalloc((void**)&d_arr_out, size * sizeof(int));
    cudaMemset(d_arr_out, 0, size * sizeof(int));

    int* d_reduce;
    cudaMalloc((void**)&d_reduce, size * sizeof(int));
    cudaMemset(d_reduce, 0, size * sizeof(int));
    cudaMemcpy(d_reduce, arr, size * sizeof(int), cudaMemcpyHostToDevice);

    int* d_total;
    cudaMalloc((void**)&d_total, sizeof(int));
    cudaMemset(d_total, 0, sizeof(int));

    int* d_blocks_sum;
    cudaMalloc((void**)&d_blocks_sum, 4 * grid_size * sizeof(int));
    cudaMemset(d_blocks_sum, 0, 4 * grid_size * sizeof(int));

    int* d_prefix_sum;
    cudaMalloc((void**)&d_prefix_sum, size * sizeof(int));
    cudaMemset(d_prefix_sum, 0, size * sizeof(int));

    int* d_scan_blocks_sum;
    cudaMalloc((void**)&d_scan_blocks_sum, size * sizeof(int));
    cudaMemset(d_scan_blocks_sum, 0, size * sizeof(int));

    int* d_global_counter;
    cudaMalloc((void**)&d_global_counter, sizeof(int));
    
    int* d_block_states;
    cudaMalloc((void**)&d_block_states, grid_size * sizeof(int));

    int* d_blocksP;
    cudaMalloc((void**)&d_blocksP, grid_size * sizeof(int));

    int* d_blocksA;
    cudaMalloc((void**)&d_blocksA, grid_size * sizeof(int));

    int shared_mem_size = block_size * sizeof(int);

    for (unsigned int bit = 0; bit <= 30; bit += 2)
    {
        // Perform order checking on the current stage
        order_checking<<<grid_size, block_size, shared_mem_size>>>(d_reduce, d_total, size);
        cudaDeviceSynchronize();

        // Check if the array is already sorted
        int total;
        cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost);
        if (total == 0)
        {
            std::cout << "Array is already sorted" << std::endl;
            break;
        }

        radix_sort<<<grid_size, block_size, shared_memory_size>>>(d_arr_in, d_blocks_sum, d_prefix_sum, bit, d_arr_out, size);
        cudaDeviceSynchronize();

        cudaMemset(d_global_counter, 0, sizeof(int));
        cudaMemset(d_block_states, 0, grid_size * sizeof(int));
        cudaMemset(d_blocksP, 0, grid_size * sizeof(int));
        cudaMemset(d_blocksA, 0, grid_size * sizeof(int));

        decoupled_lookback_scan<<<grid_size, block_size, sizeof(int)>>>(d_blocks_sum, d_global_counter, d_blocksA, d_blocksP, d_block_states, 4 * grid_size);
        cudaDeviceSynchronize();

        shift_buffer<<<grid_size, block_size>>>(d_blocks_sum, d_scan_blocks_sum, 4 * grid_size);
        cudaDeviceSynchronize();
        
        compute_new_position<<<grid_size, block_size>>>(d_arr_out, d_arr_in, d_prefix_sum, d_scan_blocks_sum, bit, size);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(arr, d_arr_in, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr_in);
    cudaFree(d_blocks_sum);
    cudaFree(d_prefix_sum);
    cudaFree(d_scan_blocks_sum);
    cudaFree(d_reduce);
    cudaFree(d_total);
    cudaFree(d_arr_out);
    cudaFree(d_global_counter);
    cudaFree(d_block_states);
    cudaFree(d_blocksP);
    cudaFree(d_blocksA);
}