#pragma once

#include "core/cubemap.hpp"
#include "core/texture.hpp"
#include "io/image.hpp"

namespace ibl {

class EquirectToCube {
public:
  EquirectToCube(int faceSize = 512);
  ~EquirectToCube() = default;

  void process(const io::Image &hdrImage, core::Cubemap &outCubemap);

private:
  int faceSize_;
};

} // namespace ibl
