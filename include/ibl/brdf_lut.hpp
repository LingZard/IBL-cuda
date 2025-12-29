#pragma once

#include "core/cuda_buffer.hpp"
#include "io/image.hpp"

namespace ibl {

class BRDFLUT {
public:
  BRDFLUT(int size = 512);
  ~BRDFLUT() = default;

  void process(io::Image &outLutImage);

private:
  int size_;
};

} // namespace ibl
