#include "core/texture.hpp"
#include "ibl/diffuse_irradiance.hpp"
#include "utils/cuda_utils.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace ibl {

__global__ void diffuseIrradianceKernel(cudaTextureObject_t envMetadata,
                                        float *outIrradianceData,
                                        int faceSize) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int face = blockIdx.z;

  if (x >= faceSize || y >= faceSize)
    return;

  float3 dir = utils::cubeCoordToWorldDir(x, y, face, faceSize);

  unsigned int seed = 1234;
  unsigned int seq = (face * faceSize * faceSize) + (y * faceSize) + x;
  curandState state;
  curand_init(seed, seq, 0, &state);

  float3 irradiance = make_float3(0.0f, 0.0f, 0.0f);
  const int SAMPLE_COUNT = 512;

  for (int i = 0; i < SAMPLE_COUNT; i++) {
    float3 sampleDir = utils::sampleHemisphereCosine(dir, &state);

    float4 sampleColor =
        texCubemap<float4>(envMetadata, sampleDir.x, sampleDir.y, sampleDir.z);

    irradiance += make_float3(sampleColor);
  }

  const float PI = 3.1415926535f;
  irradiance = irradiance * (PI / (float)SAMPLE_COUNT);

  int pixelIdx = (face * faceSize * faceSize + y * faceSize + x);
  ((float4 *)outIrradianceData)[pixelIdx] =
      make_float4(irradiance.x, irradiance.y, irradiance.z, 1.0f);
}

DiffuseIrradiance::DiffuseIrradiance(int faceSize) : faceSize_(faceSize) {}

void DiffuseIrradiance::process(const core::Cubemap &inputCubemap,
                                core::Cubemap &outIrradianceMap) {
  if (outIrradianceMap.faceSize != faceSize_) {
    outIrradianceMap = core::Cubemap(faceSize_, 4);
  }

  // 1. Prepare TextureCube from input cubemap
  std::vector<float> h_inputData(inputCubemap.faceSize * inputCubemap.faceSize *
                                 4 * 6);
  inputCubemap.data.download(h_inputData.data(), h_inputData.size());

  core::TextureCube envTex;
  envTex.create(inputCubemap.faceSize, inputCubemap.channels,
                h_inputData.data());

  // 2. Launch Kernel
  dim3 block(16, 16, 1);
  dim3 grid((faceSize_ + block.x - 1) / block.x,
            (faceSize_ + block.y - 1) / block.y, 6);

  diffuseIrradianceKernel<<<grid, block>>>(
      envTex.get(), outIrradianceMap.data.data(), faceSize_);

  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA(cudaDeviceSynchronize());
}

} // namespace ibl
