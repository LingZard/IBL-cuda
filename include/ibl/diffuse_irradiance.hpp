#pragma once

#include "core/cubemap.hpp"

namespace ibl {

class DiffuseIrradiance {
public:
  DiffuseIrradiance(int faceSize = 32);
  ~DiffuseIrradiance() = default;

  void process(const core::Cubemap &inputCubemap,
               core::Cubemap &outIrradianceMap);

private:
  int faceSize_;
};

} // namespace ibl
