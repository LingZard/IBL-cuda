#include "ibl/brdf_lut.hpp"
#include "utils/cuda_utils.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace ibl {

__global__ void brdfLutKernel(float *outData, int size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= size || y >= size)
    return;

  // The LUT is indexed by:
  // x: NdotV (from 0 to 1)
  // y: roughness (from 0 to 1)
  float mu = (float)(x + 0.5f) / (float)size;
  float roughness = (float)(y + 0.5f) / (float)size;

  // TODO: Implement BRDF LUT Integration
  // 1. Calculate View vector V from mu
  // 2. Perform Importance Sampling over the hemisphere
  // 3. Calculate Geometry function and Fresnel terms
  // 4. Output RG result: float2(Scale, Bias)

  float2 res = make_float2(mu, roughness); // Placeholder

  int pixelIdx = (y * size + x);
  ((float2 *)outData)[pixelIdx] = res;
}

BRDFLUT::BRDFLUT(int size) : size_(size) {}

void BRDFLUT::process(io::Image &outLutImage) {
  outLutImage.width = size_;
  outLutImage.height = size_;
  outLutImage.channels = 2; // Output is RG (Scale, Bias)
  outLutImage.data.resize(size_ * size_ * 2);

  core::CudaBuffer<float> d_lutData(size_ * size_ * 2);

  dim3 block(16, 16);
  dim3 grid((size_ + block.x - 1) / block.x, (size_ + block.y - 1) / block.y);

  brdfLutKernel<<<grid, block>>>(d_lutData.data(), size_);

  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  d_lutData.download(outLutImage.data.data(), outLutImage.data.size());
}

} // namespace ibl
