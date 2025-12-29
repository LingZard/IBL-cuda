#pragma once

#include <cuda_runtime.h>

__device__ __inline__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __inline__ float3 &operator+=(float3 &a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

__device__ __inline__ float3 operator*(float3 a, float s) {
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

// Helper: Convert float4 to float3
__device__ __inline__ float3 make_float3(float4 f) {
  return make_float3(f.x, f.y, f.z);
}
