#pragma once

#include "error.cuh"

template <typename T>
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

    void copyFromHost(const T* src, int size)
    {
        cudaXMemcpy(data_, src, size * sizeof(T), cudaMemcpyHostToDevice);
    }

    void copyToHost(T* dst, int size) const
    {
        cudaXMemcpy(dst, data_, size * sizeof(T), cudaMemcpyDeviceToHost);
    }

    void allocate(int size)
    {
        cudaXMalloc((void**)&data_, size * sizeof(T));
    }

    void setTo(int size, int value)
    {
        cudaXMemset(data_, value, size * sizeof(T));
    }

    void setTo(int offset, int size, int value)
    {
        cudaXMemset(data_ + offset, value, size * sizeof(T));
    }

    void free()
    {
        if (data_ != nullptr)
        {
            cudaXFree(data_);
            data_ = nullptr;
        }
    }

    T* data_;
};