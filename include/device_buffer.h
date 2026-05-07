#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <utility>
#include <vector>


template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t n) : n_(n) {
        cudaMalloc(&ptr_, n_ * sizeof(T));
    }

    explicit DeviceBuffer(const std::vector<T> &host) : DeviceBuffer(host.size()) {
        cudaMemcpy(ptr_, host.data(), n_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    ~DeviceBuffer() { cudaFree(ptr_); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&o) noexcept : ptr_(o.ptr_), n_(o.n_) {
        o.ptr_ = nullptr;
        o.n_ = 0;
    }
    DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
        if (this != &o) {
            cudaFree(ptr_);
            ptr_ = o.ptr_;
            n_ = o.n_;
            o.ptr_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }

    T *data() { return ptr_; }
    const T *data() const { return ptr_; }
    std::size_t size() const { return n_; }

    std::vector<T> to_host() const {
        std::vector<T> out(n_);
        cudaMemcpy(out.data(), ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost);
        return out;
    }

  private:
    T *ptr_ = nullptr;
    std::size_t n_ = 0;
};
