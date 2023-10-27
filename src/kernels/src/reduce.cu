#include "reduce.cuh"

///////////////////////// baseline reduce ///////////////////////////

__global__ void kernel_reduce_baseline(const int* __restrict__ buffer, int* __restrict__ total, int size)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size)
        atomicAdd(&total[0], buffer[id]);
}

///////////////////////// reduce 1 ///////////////////////////

__global__ void reduce1(const int* __restrict__ buffer, int* __restrict__ total, int size)
{
    extern __shared__ int shared[];
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size)
        shared[threadIdx.x] = buffer[id];
    else
        shared[threadIdx.x] = 0;
    
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (threadIdx.x % (2 * s) == 0)
            shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&total[0], shared[0]);
}

///////////////////////// reduce 2 ///////////////////////////

__global__ void reduce2(const int* __restrict__ buffer, int* __restrict__ total, int size)
{
    extern __shared__ int shared[];
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size)
        shared[threadIdx.x] = buffer[id];
    else
        shared[threadIdx.x] = 0;
    
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * threadIdx.x;
        if (index < blockDim.x)
            shared[index] += shared[index + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&total[0], shared[0]);
}

///////////////////////// reduce 3 ///////////////////////////

__global__ void reduce3(const int* __restrict__ buffer, int* __restrict__ total, int size)
{
    extern __shared__ int shared[];
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size)
        shared[threadIdx.x] = buffer[id];
    else
        shared[threadIdx.x] = 0;
    
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&total[0], shared[0]);
}

///////////////////////// reduce 4 ///////////////////////////

__global__ void reduce4(const int* __restrict__ buffer, int* __restrict__ total, int size)
{
    extern __shared__ int shared[];
    const int id = threadIdx.x + blockIdx.x * (blockDim.x * 2);
    if (id < size)
        shared[threadIdx.x] = buffer[id] + buffer[id + blockDim.x];
    else
        shared[threadIdx.x] = 0;
    
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&total[0], shared[0]);
}

///////////////////////// reduce 5 ///////////////////////////

__device__ void warp_reduce5(int* shared, int gid)
{
    shared[gid] += shared[gid + 32]; __syncwarp();
    shared[gid] += shared[gid + 16]; __syncwarp();
    shared[gid] += shared[gid + 8]; __syncwarp();
    shared[gid] += shared[gid + 4]; __syncwarp();
    shared[gid] += shared[gid + 2]; __syncwarp();
    shared[gid] += shared[gid + 1]; __syncwarp();
}

__global__ void reduce5(const int* __restrict__ buffer, int* __restrict__ total, int size)
{
    extern __shared__ int shared[];
    const int id = threadIdx.x + blockIdx.x * (blockDim.x * 2);
    if (id < size)
        shared[threadIdx.x] = buffer[id] + buffer[id + blockDim.x];
    else
        shared[threadIdx.x] = 0;
    
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (threadIdx.x < s)
            shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x < 32)
        warp_reduce5(shared, threadIdx.x);

    if (threadIdx.x == 0)
        atomicAdd(&total[0], shared[0]);
}

///////////////////////// reduce 6 ///////////////////////////

__device__ void warp_reduce6(int* shared, int gid, int blockSize)
{
    if (blockSize >= 64) shared[gid] += shared[gid + 32]; __syncwarp();
    if (blockSize >= 32) shared[gid] += shared[gid + 16]; __syncwarp();
    if (blockSize >= 16) shared[gid] += shared[gid + 8]; __syncwarp();
    if (blockSize >= 8) shared[gid] += shared[gid + 4]; __syncwarp();
    if (blockSize >= 4) shared[gid] += shared[gid + 2]; __syncwarp();
    if (blockSize >= 2) shared[gid] += shared[gid + 1]; __syncwarp();
}

__global__ void reduce6(const int* __restrict__ buffer, int* __restrict__ total, int blockSize, int size)
{
    extern __shared__ int shared[];
    const int id = threadIdx.x + blockIdx.x * (blockDim.x * 2);
    if (id < size)
        shared[threadIdx.x] = buffer[id] + buffer[id + blockDim.x];
    else
        shared[threadIdx.x] = 0;
    
    __syncthreads();

    if (blockSize >= 1024) { if (threadIdx.x < 512) { shared[threadIdx.x] += shared[threadIdx.x + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (threadIdx.x < 256) { shared[threadIdx.x] += shared[threadIdx.x + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (threadIdx.x < 128) { shared[threadIdx.x] += shared[threadIdx.x + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (threadIdx.x < 64) { shared[threadIdx.x] += shared[threadIdx.x + 64]; } __syncthreads(); }
    
    if (threadIdx.x < 32)
        warp_reduce6(shared, threadIdx.x, blockSize);

    if (threadIdx.x == 0)
        atomicAdd(&total[0], shared[0]);
}

///////////////////////// reduce 7 ///////////////////////////

__device__ void warp_reduce7(int* shared, int gid, int blockSize)
{
    if (blockSize >= 64) shared[gid] += shared[gid + 32]; __syncwarp();
    if (blockSize >= 32) shared[gid] += shared[gid + 16]; __syncwarp();
    if (blockSize >= 16) shared[gid] += shared[gid + 8]; __syncwarp();
    if (blockSize >= 8) shared[gid] += shared[gid + 4]; __syncwarp();
    if (blockSize >= 4) shared[gid] += shared[gid + 2]; __syncwarp();
    if (blockSize >= 2) shared[gid] += shared[gid + 1]; __syncwarp();
}

__global__ void reduce7(const int* __restrict__ buffer, int* __restrict__ total, int blockSize, int size)
{
    extern __shared__ int shared[];
    int id = threadIdx.x + blockIdx.x * (blockDim.x * 2);
    if (id < size)
    {
        shared[threadIdx.x] = 0;
        while (id < size)
        {
            shared[threadIdx.x] += buffer[id] + buffer[id + blockDim.x];
            id += blockDim.x * 2 * gridDim.x;
        }
    }   
    else
        shared[threadIdx.x] = 0;
    
    __syncthreads();

    if (blockSize >= 1024) { if (threadIdx.x < 512) { shared[threadIdx.x] += shared[threadIdx.x + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (threadIdx.x < 256) { shared[threadIdx.x] += shared[threadIdx.x + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (threadIdx.x < 128) { shared[threadIdx.x] += shared[threadIdx.x + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (threadIdx.x < 64) { shared[threadIdx.x] += shared[threadIdx.x + 64]; } __syncthreads(); }
    
    if (threadIdx.x < 32)
        warp_reduce7(shared, threadIdx.x, blockSize);

    if (threadIdx.x == 0)
        atomicAdd(&total[0], shared[0]);
}

///////////////////////// reduce 8 ///////////////////////////

__inline__ __device__ int warp_reduce8(int value)
{
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        value += __shfl_down_sync(~0, value, offset);
    return value;
}

__inline__ __device__ int blockReduceSum(int value)
{
    static __shared__ int shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    value = warp_reduce8(value);

    if (lane == 0)
        shared[wid] = value;

    __syncthreads();

    value = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0)
        value = warp_reduce8(value);

    return value;
}

__global__ void reduce8(const int* __restrict__ buffer, int* __restrict__ total, int size)
{
    int id = threadIdx.x + blockIdx.x * (blockDim.x * 2);
    if (id < size)
    {
        int sum = 0;

        while (id < size)
        {
            sum += buffer[id] + buffer[id + blockDim.x];
            id += blockDim.x * 2 * gridDim.x;
        }

        sum = blockReduceSum(sum);

        if ((threadIdx.x & (warpSize - 1)) == 0)
            atomicAdd(&total[0], sum);
    }
}

///////////////////////// reduce 9 ///////////////////////////

__inline__ __device__ int warp_reduce9(int value)
{
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        value += __shfl_down_sync(~0, value, offset);
    return value;
}

__global__ void reduce9(const int* __restrict__ buffer, int* __restrict__ total, int size)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size)
    {
        int sum = 0;

        while (id < (size / 4))
        {
            int4 data = reinterpret_cast<const int4*>(buffer)[id];
            sum += data.x + data.y + data.z + data.w;
            id += blockDim.x * gridDim.x;
        }

        sum = warp_reduce9(sum);

        if ((threadIdx.x & (warpSize - 1)) == 0)
            atomicAdd(&total[0], sum);
    }
}