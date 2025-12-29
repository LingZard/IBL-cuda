#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::string msg =                                                        \
          "CUDA Error: " + std::string(cudaGetErrorString(err)) + " at " +     \
          __FILE__ + ":" + std::to_string(__LINE__);                           \
      std::cerr << msg << std::endl;                                           \
      throw std::runtime_error(msg);                                           \
    }                                                                          \
  } while (0)

#define CHECK_LAST_CUDA_ERROR() CHECK_CUDA(cudaGetLastError())
