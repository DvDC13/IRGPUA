#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#define cudaCheckError()                                             \
    {                                                                \
        cudaError_t e = cudaGetLastError();                          \
        if (e != cudaSuccess)                                        \
        {                                                            \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                           \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

#define gpu_err_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

        if (abort)
            exit(code);
    }
}

inline void cudaXMalloc(void **ptr, size_t size)
{
    gpu_err_check(cudaMalloc(ptr, size));
}

inline void cudaXMemset(void *ptr, int value, size_t count)
{
    gpu_err_check(cudaMemset(ptr, value, count));
}

inline void cudaXFree(void *ptr)
{
    gpu_err_check(cudaFree(ptr));
}

inline void cudaXMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind)
{
    gpu_err_check(cudaMemcpy(dst, src, size, kind));
}

inline void cudaXDeviceSynchronize()
{
    gpu_err_check(cudaDeviceSynchronize());
}

inline void cudaXDeviceReset()
{
    gpu_err_check(cudaDeviceReset());
}

inline void cudaXMallocHost(void **ptr, size_t size)
{
    gpu_err_check(cudaMallocHost(ptr, size));
}

inline void cudaXFreeHost(void *ptr)
{
    gpu_err_check(cudaFreeHost(ptr));
}

inline void cudaXMemcpyAsync(void *dst, const void *src, size_t size, cudaMemcpyKind kind, cudaStream_t stream)
{
    gpu_err_check(cudaMemcpyAsync(dst, src, size, kind, stream));
}

inline void cudaXStreamCreate(cudaStream_t *stream)
{
    gpu_err_check(cudaStreamCreate(stream));
}

inline void cudaXStreamDestroy(cudaStream_t stream)
{
    gpu_err_check(cudaStreamDestroy(stream));
}

inline void cudaXStreamSynchronize(cudaStream_t stream)
{
    gpu_err_check(cudaStreamSynchronize(stream));
}