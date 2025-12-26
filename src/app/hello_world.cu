#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

static inline void checkCuda(cudaError_t err, const char* expr, const char* file, int line) {
  if (err == cudaSuccess) return;
  fprintf(stderr, "CUDA error: %s\n  expr: %s\n  at: %s:%d\n",
          cudaGetErrorString(err), expr, file, line);
  exit(1);
}

#define CHECK_CUDA(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

__global__ void helloKernel() {
  printf("Hello from CUDA kernel! block=(%d,%d,%d) thread=(%d,%d,%d)\n",
         (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z,
         (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z);
}

int main() {
  int deviceCount = 0;
  CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
  if (deviceCount <= 0) {
    fprintf(stderr, "No CUDA devices found.\n");
    return 2;
  }

  int device = 0;
  CHECK_CUDA(cudaSetDevice(device));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  printf("Using GPU %d: %s (cc %d.%d)\n", device, prop.name, prop.major, prop.minor);

  printf("Launching kernel...\n");
  helloKernel<<<1, 1>>>();
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  printf("Done.\n");
  return 0;
}


