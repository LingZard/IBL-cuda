#include "ibl/brdf_lut.hpp"
#include "utils/cuda_utils.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace ibl {

__device__ __inline__ float geometrySmithGGX(float NoV, float NoL,
                                             float roughness) {
  // IBL: k = (alpha^2) / 2
  // float alpha = roughness * roughness;
  float k = (roughness * roughness) / 2.0f;

  float G_v = NoV / (NoV * (1.0f - k) + k);
  float G_l = NoL / (NoL * (1.0f - k) + k);
  return G_v * G_l;
}

__global__ void brdfLutKernel(float *outData, int size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= size || y >= size)
    return;

  // The LUT is indexed by:
  // x: NdotV (from 0 to 1)
  // y: roughness (from 1 to 0), matching standard UV expectations (v=0 at
  // bottom)
  float NoV = (float)(x + 0.5f) / (float)size;
  float roughness = (float)(size - 1 - y + 0.5f) / (float)size;

  float3 V = make_float3(sqrtf(1.0f - NoV * NoV), 0.0f, NoV); // Isotropic
  float3 N = make_float3(0.0f, 0.0f, 1.0f);

  unsigned int seed = 1234;
  unsigned int seq = (y * size) + x;
  curandState state;
  curand_init(seed, seq, 0, &state);

  const int SAMPLE_COUNT = 512;

  float2 brdf = make_float2(0.0f, 0.0f);
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    float3 H = utils::sampleGGX(N, roughness, &state);
    float3 L = normalize(2.0f * dot(V, H) * H - V);
    float NoL = dot(N, L);
    float VoH = dot(V, H);
    float NoH = dot(N, H);

    if (NoL > 0.0f) {
      float G = geometrySmithGGX(NoV, NoL, roughness);
      float G_vis = G * (VoH) / (NoH * NoV);

      float F_k = powf(1.0f - VoH, 5);
      brdf.x += G_vis * (1 - F_k);
      brdf.y += G_vis * F_k;
    }
  }
  brdf.x /= (float)SAMPLE_COUNT;
  brdf.y /= (float)SAMPLE_COUNT;

  int pixelIdx = (y * size + x);
  ((float3 *)outData)[pixelIdx] = make_float3(brdf.x, brdf.y, 0.0f);
}

BRDFLUT::BRDFLUT(int size) : size_(size) {}

void BRDFLUT::process(io::Image &outLutImage) {
  outLutImage.width = size_;
  outLutImage.height = size_;
  outLutImage.channels = 3; // Output as RGB for correct PNG visualization
  outLutImage.data.resize(size_ * size_ * 3);

  core::CudaBuffer<float> d_lutData(size_ * size_ * 3);

  dim3 block(16, 16);
  dim3 grid((size_ + block.x - 1) / block.x, (size_ + block.y - 1) / block.y);

  brdfLutKernel<<<grid, block>>>(d_lutData.data(), size_);

  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());

  d_lutData.download(outLutImage.data.data(), outLutImage.data.size());
}

} // namespace ibl
