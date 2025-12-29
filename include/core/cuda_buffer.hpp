#pragma once

#include "utils/cuda_utils.hpp"
#include <cuda_runtime.h>

namespace core {

template <typename T> class CudaBuffer {
public:
  CudaBuffer() : d_ptr_(nullptr), size_(0) {}

  explicit CudaBuffer(size_t count) : d_ptr_(nullptr), size_(count) {
    allocate(count);
  }

  ~CudaBuffer() { free(); }

  // Disable copy
  CudaBuffer(const CudaBuffer &) = delete;
  CudaBuffer &operator=(const CudaBuffer &) = delete;

  // Move
  CudaBuffer(CudaBuffer &&other) noexcept
      : d_ptr_(other.d_ptr_), size_(other.size_) {
    other.d_ptr_ = nullptr;
    other.size_ = 0;
  }

  CudaBuffer &operator=(CudaBuffer &&other) noexcept {
    if (this != &other) {
      free();
      d_ptr_ = other.d_ptr_;
      size_ = other.size_;
      other.d_ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void allocate(size_t count) {
    free();
    size_ = count;
    if (size_ > 0) {
      CHECK_CUDA(cudaMalloc(&d_ptr_, size_ * sizeof(T)));
    }
  }

  void free() {
    if (d_ptr_) {
      cudaFree(d_ptr_);
      d_ptr_ = nullptr;
    }
    size_ = 0;
  }

  void upload(const T *h_ptr, size_t count) {
    if (count > size_) {
      allocate(count);
    }
    CHECK_CUDA(
        cudaMemcpy(d_ptr_, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  void download(T *h_ptr, size_t count) const {
    size_t copy_count = (count < size_) ? count : size_;
    CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr_, copy_count * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

  T *data() { return d_ptr_; }
  const T *data() const { return d_ptr_; }
  size_t size() const { return size_; }

private:
  T *d_ptr_;
  size_t size_;
};

} // namespace core
