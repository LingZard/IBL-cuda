#include "ibl/equirect_to_cube.hpp"
#include "utils/cuda_utils.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

namespace ibl {

template <int CHANNELS>
__global__ void equirectToCubeKernel(cudaTextureObject_t equirectTex,
                                     float *outCubemapData, int faceSize) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int face = blockIdx.z;

  if (x >= faceSize || y >= faceSize)
    return;

  float3 dir = utils::cubeCoordToWorldDir(x, y, face, faceSize);

  const float PI = 3.1415926535f;
  float theta = atan2f(dir.z, dir.x);
  float phi = acosf(dir.y);

  float sample_u = (theta / (2.0f * PI)) + 0.5f;
  float sample_v = phi / PI;

  float4 color = tex2D<float4>(equirectTex, sample_u, sample_v);

  int pixelIdx = (face * faceSize * faceSize + y * faceSize + x);
  if (CHANNELS == 4) {
    // Vectorized write for float4 alignment
    ((float4 *)outCubemapData)[pixelIdx] = color;
  } else {
    int base = pixelIdx * CHANNELS;
    if (CHANNELS >= 1)
      outCubemapData[base + 0] = color.x;
    if (CHANNELS >= 2)
      outCubemapData[base + 1] = color.y;
    if (CHANNELS >= 3)
      outCubemapData[base + 2] = color.z;
  }
}

EquirectToCube::EquirectToCube(int faceSize) : faceSize_(faceSize) {}

void EquirectToCube::process(const io::Image &hdrImage,
                             core::Cubemap &outCubemap) {
  // 1. Prepare output cubemap if not already allocated
  if (outCubemap.faceSize != faceSize_) {
    outCubemap =
        core::Cubemap(faceSize_, 4); // Use 4 channels for alignment/simplicity
  }

  // 2. Prepare input texture
  core::Texture2D hdrTex;
  hdrTex.create(hdrImage.width, hdrImage.height, hdrImage.channels,
                hdrImage.data.data());

  // 3. Launch Kernel
  dim3 block(16, 16, 1);
  dim3 grid((faceSize_ + block.x - 1) / block.x,
            (faceSize_ + block.y - 1) / block.y, 6);

  if (outCubemap.channels == 4) {
    equirectToCubeKernel<4><<<grid, block>>>(
        hdrTex.get(), outCubemap.baseLevelData().data(), faceSize_);
  } else if (outCubemap.channels == 1) {
    equirectToCubeKernel<1><<<grid, block>>>(
        hdrTex.get(), outCubemap.baseLevelData().data(), faceSize_);
  } else if (outCubemap.channels == 2) {
    equirectToCubeKernel<2><<<grid, block>>>(
        hdrTex.get(), outCubemap.baseLevelData().data(), faceSize_);
  } else {
    std::cerr << "Unsupported channel count for EquirectToCube" << std::endl;
  }

  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());
}

} // namespace ibl
