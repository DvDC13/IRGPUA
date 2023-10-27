#include "radix_sort.cuh"

/* Based on this article
    https://www.semanticscholar.org/paper/Fast-4-way-parallel-radix-sorting-on-GPUs-Ha-Kr%C3%BCger/eaa887377239049f2f6d55f23830ce5f2bb6f38c
*/

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

        __syncthreads();
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

    // Shift to the right to produce an exclusive prefix sum
    s_mask_scan[tid + 1] = s_mask_scan[tid];

    if (tid == 0) s_mask_scan[0] = 0;

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

    DeviceArray d_arr_in(size, 0);
    d_arr_in.copyFromHost(arr, size);

    DeviceArray d_arr_out(size, 0);

    DeviceArray d_reduce(size, 0);
    d_reduce.copyFromHost(arr, size);

    DeviceArray d_total(1, 0);

    DeviceArray d_blocks_sum(4 * grid_size, 0);

    DeviceArray d_prefix_sum(size, 0);

    DeviceArray d_scan_blocks_sum(size, 0);

    DeviceArray d_global_counter(1, 0);

    DeviceArray d_blocks_aggregate(grid_size, 0);
    
    cuda::std::atomic<char>* d_block_states;
    cudaXMalloc((void**)&d_block_states, grid_size * sizeof(cuda::std::atomic<int>));

    int shared_mem_size = block_size * sizeof(int);

    for (unsigned int bit = 0; bit <= 30; bit += 2)
    {
        // Perform order checking on the current stage
        order_checking<<<grid_size, block_size, shared_mem_size>>>(d_reduce.data_, d_total.data_, size);
        cudaDeviceSynchronize();

        // Check if the array is already sorted
        int total;
        cudaMemcpy(&total, d_total.data_, sizeof(int), cudaMemcpyDeviceToHost);
        if (total == 0)
        {
            std::cout << "Array is already sorted" << std::endl;
            break;
        }

        // Perform radix sort
        radix_sort<<<grid_size, block_size, shared_memory_size>>>(d_arr_in.data_, d_blocks_sum.data_, d_prefix_sum.data_, bit, d_arr_out.data_, size);
        cudaDeviceSynchronize();

        d_global_counter.setTo(1, 0);
        d_blocks_aggregate.setTo(grid_size, 0);
        cudaXMemset(d_block_states, 'X', grid_size * sizeof(char));

        // Perform scan on the blocks sum
        decoupled_look_back_optimized<<<grid_size, block_size, sizeof(int)>>>(d_blocks_sum.data_, 4 * grid_size, d_blocks_aggregate.data_, d_global_counter.data_, d_block_states);
        cudaDeviceSynchronize();

        // Shift for an exclusive prefix sum
        shift_buffer<<<grid_size, block_size>>>(d_blocks_sum.data_, d_scan_blocks_sum.data_, 4 * grid_size);
        cudaDeviceSynchronize();
        
        // Compute the new position of each element
        compute_new_position<<<grid_size, block_size>>>(d_arr_out.data_, d_arr_in.data_, d_prefix_sum.data_, d_scan_blocks_sum.data_, bit, size);
        cudaDeviceSynchronize();
    }

    d_arr_in.copyToHost(arr, size);

    cudaXFree(d_block_states);
}