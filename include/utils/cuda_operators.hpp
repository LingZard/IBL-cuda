#pragma once

#include <cmath>
#include <cuda_runtime.h>

#if !defined(__CUDACC__) && !defined(__NVCC__)
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
#endif

__device__ __inline__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __inline__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __inline__ float3 &operator+=(float3 &a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

__device__ __inline__ float3 &operator+=(float3 &a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

__device__ __inline__ float3 operator*(float3 a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __inline__ float3 operator*(float s, float3 a) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __inline__ float3 operator/(float3 a, float s) {
  float invS = 1.0f / s;
  return make_float3(a.x * invS, a.y * invS, a.z * invS);
}

__device__ __inline__ float3 &operator/=(float3 &a, float s) {
  float invS = 1.0f / s;
  a.x *= invS;
  a.y *= invS;
  a.z *= invS;
  return a;
}

__device__ __inline__ float4 operator*(float4 a, float s) {
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

__device__ __inline__ float4 operator*(float s, float4 a) {
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

__device__ __inline__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __inline__ float3 normalize(float3 v) {
  float invLen = rsqrtf(dot(v, v));
  return v * invLen;
}

// Helper: Convert float4 to float3
__device__ __inline__ float3 make_float3(float4 f) {
  return make_float3(f.x, f.y, f.z);
}
