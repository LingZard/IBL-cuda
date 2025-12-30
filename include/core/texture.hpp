#pragma once

#include "utils/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace core {

class Texture2D {
public:
  Texture2D() : texObj(0), cuArray(nullptr) {}
  ~Texture2D() { destroy(); }

  void create(int width, int height, int channels, const float *h_data) {
    destroy();

    int targetChannels = channels;
    const float *finalData = h_data;
    std::vector<float> paddedData;

    if (channels == 3) {
      std::cout << "[Texture2D] Warning: 3-channel HDR detected. Implicitly "
                   "padding to 4 channels (float4) for better GPU alignment."
                << std::endl;
      targetChannels = 4;
      paddedData.resize(width * height * 4);
      for (int i = 0; i < width * height; ++i) {
        paddedData[i * 4 + 0] = h_data[i * 3 + 0];
        paddedData[i * 4 + 1] = h_data[i * 3 + 1];
        paddedData[i * 4 + 2] = h_data[i * 3 + 2];
        paddedData[i * 4 + 3] = 1.0f; // Alpha = 1.0
      }
      finalData = paddedData.data();
    }

    cudaChannelFormatDesc channelDesc;
    if (targetChannels == 1)
      channelDesc = cudaCreateChannelDesc<float>();
    else if (targetChannels == 2)
      channelDesc = cudaCreateChannelDesc<float2>();
    else if (targetChannels == 4)
      channelDesc = cudaCreateChannelDesc<float4>();
    else {
      throw std::runtime_error("Only 1, 2, 3, or 4 channels supported for "
                               "Texture2D. (3-ch will be padded to 4-ch)");
    }

    CHECK_CUDA(cudaMallocArray(&cuArray, &channelDesc, width, height));
    CHECK_CUDA(cudaMemcpy2DToArray(cuArray, 0, 0, finalData,
                                   width * targetChannels * sizeof(float),
                                   width * targetChannels * sizeof(float),
                                   height, cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
  }

  void destroy() {
    if (texObj) {
      cudaDestroyTextureObject(texObj);
      texObj = 0;
    }
    if (cuArray) {
      cudaFreeArray(cuArray);
      cuArray = nullptr;
    }
  }

  cudaTextureObject_t get() const { return texObj; }

private:
  cudaTextureObject_t texObj;
  cudaArray_t cuArray;
};

class TextureCube {
public:
  TextureCube() : texObj(0), cuArray(nullptr) {}
  ~TextureCube() { destroy(); }

  void create(int faceSize, int channels, const float *h_data) {
    destroy();

    if (channels != 4) {
      throw std::runtime_error("TextureCube currently only supports 4 channels "
                               "(float4) for simplicity.");
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaExtent extent = make_cudaExtent(faceSize, faceSize, 6);

    CHECK_CUDA(
        cudaMalloc3DArray(&cuArray, &channelDesc, extent, cudaArrayCubemap));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(
        (void *)h_data, faceSize * sizeof(float4), faceSize, faceSize);
    copyParams.dstArray = cuArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    CHECK_CUDA(cudaMemcpy3D(&copyParams));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
  }

  void destroy() {
    if (texObj) {
      cudaDestroyTextureObject(texObj);
      texObj = 0;
    }
    if (cuArray) {
      cudaFreeArray(cuArray);
      cuArray = nullptr;
    }
  }

  cudaTextureObject_t get() const { return texObj; }

private:
  cudaTextureObject_t texObj;
  cudaArray_t cuArray;
};

} // namespace core
