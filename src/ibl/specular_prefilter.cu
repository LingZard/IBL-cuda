#include "core/texture.hpp"
#include "ibl/specular_prefilter.hpp"
#include "utils/cuda_utils.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace ibl {

__global__ void specularPrefilterKernel(cudaTextureObject_t envMetadata,
                                        float *outLevelData, int faceSize,
                                        float roughness) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int face = blockIdx.z;

  if (x >= faceSize || y >= faceSize)
    return;

  float3 dir = utils::cubeCoordToWorldDir(x, y, face, faceSize);

  float3 N = dir;
  float3 R = dir;
  float3 V = dir;

  unsigned int seed = 1234;
  unsigned int seq = (face * faceSize * faceSize) + (y * faceSize) + x;
  curandState state;
  curand_init(seed, seq, 0, &state);

  float3 prefilteredColor = make_float3(0.0f, 0.0f, 0.0f);
  const int SAMPLE_COUNT = 512;

  float totalWeight = 0.0f;
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    float3 H = utils::sampleGGX(dir, roughness, &state);
    float3 L = normalize(2.0f * dot(V, H) * H - V);
    float NoL = dot(N, L);
    if (NoL > 0.0f) {
      float4 sampleColor = texCubemap<float4>(envMetadata, L.x, L.y, L.z);
      prefilteredColor += NoL * sampleColor;
      totalWeight += NoL;
    }
  }

  if (totalWeight > 0.0f) {
    prefilteredColor /= totalWeight;
  } else {
    prefilteredColor =
        make_float3(texCubemap<float4>(envMetadata, N.x, N.y, N.z));
  }

  int pixelIdx = (face * faceSize * faceSize + y * faceSize + x);
  ((float4 *)outLevelData)[pixelIdx] = make_float4(
      prefilteredColor.x, prefilteredColor.y, prefilteredColor.z, 1.0f);
}

SpecularPrefilter::SpecularPrefilter(int faceSize, int maxMipLevels)
    : faceSize_(faceSize), maxMipLevels_(maxMipLevels) {}

void SpecularPrefilter::process(const core::Cubemap &envCubemap,
                                core::Cubemap &outPrefilteredMap) {
  if (outPrefilteredMap.faceSize != faceSize_ ||
      outPrefilteredMap.numLevels != maxMipLevels_) {
    outPrefilteredMap = core::Cubemap(faceSize_, 4, maxMipLevels_);
  }

  // 1. Prepare TextureCube from input cubemap
  std::vector<float> h_inputData(envCubemap.faceSize * envCubemap.faceSize * 4 *
                                 6);
  envCubemap.baseLevelData().download(h_inputData.data(), h_inputData.size());

  core::TextureCube envTex;
  envTex.create(envCubemap.faceSize, 4, h_inputData.data());

  // 2. Process each Mip Level
  for (int level = 0; level < maxMipLevels_; ++level) {
    int levelSize = faceSize_ >> level;
    if (levelSize < 1)
      levelSize = 1;

    float roughness = (float)level / (float)(maxMipLevels_ - 1);

    dim3 block(16, 16, 1);
    dim3 grid((levelSize + block.x - 1) / block.x,
              (levelSize + block.y - 1) / block.y, 6);

    specularPrefilterKernel<<<grid, block>>>(
        envTex.get(), outPrefilteredMap.levels[level].data.data(), levelSize,
        roughness);

    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());
  }
}

} // namespace ibl
