#pragma once

#include "error.cuh"


class DeviceArray
{
public:
    explicit DeviceArray(int size)
        : data_(nullptr)
    {
        allocate(size);
    }

    explicit DeviceArray(int size, int value)
        : data_(nullptr)
    {
        allocate(size);
        setTo(size, value);
    }

    ~DeviceArray()
    {
        free();
    }

    void resize(int size)
    {
        free();
        allocate(size);
    }

    void copyFromHost(const int* src, int size)
    {
        cudaXMemcpy(data_, src, size * sizeof(int), cudaMemcpyHostToDevice);
    }

    void copyToHost(int* dst, int size) const
    {
        cudaXMemcpy(dst, data_, size * sizeof(int), cudaMemcpyDeviceToHost);
    }

    void allocate(int size)
    {
        cudaXMalloc((void**)&data_, size * sizeof(int));
    }

    void setTo(int size, int value)
    {
        cudaXMemset(data_, value, size * sizeof(int));
    }

    void setTo(int offset, int size, int value)
    {
        cudaXMemset(data_ + offset, value, size * sizeof(int));
    }

    void free()
    {
        if (data_ != nullptr)
        {
            cudaXFree(data_);
            data_ = nullptr;
        }
    }

    int* data_;
};