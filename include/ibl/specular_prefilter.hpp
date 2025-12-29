#pragma once

#include "core/cubemap.hpp"

namespace ibl {

class SpecularPrefilter {
public:
  SpecularPrefilter(int faceSize = 128, int maxMipLevels = 5);
  ~SpecularPrefilter() = default;

  void process(const core::Cubemap &envCubemap,
               core::Cubemap &outPrefilteredMap);

private:
  int faceSize_;
  int maxMipLevels_;
};

} // namespace ibl
