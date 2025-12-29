#pragma once

#include "utils/cuda_operators.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
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

namespace utils {

#if !defined(__CUDACC__) && !defined(__NVCC__)
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
#endif

__device__ __inline__ float3 cubeCoordToWorldDir(int x, int y, int face,
                                                 int faceSize) {
  float u = (2.0f * (x + 0.5f) / (float)faceSize) - 1.0f;
  float v = (2.0f * (y + 0.5f) / (float)faceSize) - 1.0f;

  float3 dir;
  if (face == 0)
    dir = make_float3(1.0f, -v, -u); // +X
  else if (face == 1)
    dir = make_float3(-1.0f, -v, u); // -X
  else if (face == 2)
    dir = make_float3(u, 1.0f, v); // +Y
  else if (face == 3)
    dir = make_float3(u, -1.0f, -v); // -Y
  else if (face == 4)
    dir = make_float3(u, -v, 1.0f); // +Z
  else
    dir = make_float3(-u, -v, -1.0f); // -Z

  float invLen = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
  dir.x *= invLen;
  dir.y *= invLen;
  dir.z *= invLen;

  return dir;
}

__device__ __inline__ void buildOrthonormalBasis(const float3 n, float3 &b1,
                                                 float3 &b2) {
  float sign = copysignf(1.0f, n.z);
  const float a = -1.0f / (sign + n.z);
  const float b = n.x * n.y * a;

  b1 = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
  b2 = make_float3(b, sign + n.y * n.y * a, -n.y);
}

// TODO: low discrepancy sampling
__device__ __inline__ float3 sampleHemisphereCosine(const float3 n,
                                                    curandState *state) {
  float r1 = curand_uniform(state);
  float r2 = curand_uniform(state);

  float phi = 2.0f * 3.14159265f * r1;
  float radius = sqrtf(r2);
  float3 localRay =
      make_float3(cosf(phi) * radius, sinf(phi) * radius, sqrtf(1.0f - r2));

  float3 t, b;
  buildOrthonormalBasis(n, t, b);

  return t * localRay.x + b * localRay.y + n * localRay.z;
}

__device__ __inline__ float3 sampleGGX(const float3 n, float roughness,
                                       curandState *state) {
  // return normalized half vector H
  float a = roughness * roughness;
  float r1 = curand_uniform(state);
  float r2 = curand_uniform(state);

  float phi = 2.0f * 3.14159265f * r1;
  float cosTheta = sqrtf((1.0f - r2) / ((a * a - 1.0f) * r2 + 1.0f));
  float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
  float3 localDir =
      make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
  float3 t, b;
  buildOrthonormalBasis(n, t, b);
  return t * localDir.x + b * localDir.y + n * localDir.z;
}
} // namespace utils
