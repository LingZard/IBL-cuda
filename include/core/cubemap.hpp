#pragma once

#include "core/cuda_buffer.hpp"
#include "io/image.hpp"
#include <string>
#include <vector>

namespace core {

enum class CubeFace {
  POSIX = 0, // +X
  NEGX = 1,  // -X
  POSY = 2,  // +Y
  NEGY = 3,  // -Y
  POSZ = 4,  // +Z
  NEGZ = 5   // -Z
};

struct Cubemap {
  int faceSize;
  int channels;
  CudaBuffer<float> data; // Stores 6 faces, each faceSize * faceSize * channels

  Cubemap() : faceSize(0), channels(0) {}
  Cubemap(int size, int comps) : faceSize(size), channels(comps) {
    data.allocate(size * size * comps * 6);
  }

  void saveFaces(const std::string &prefix) const {
    std::vector<float> h_data(faceSize * faceSize * channels * 6);
    data.download(h_data.data(), h_data.size());

    const char *faceNames[] = {"posx", "negx", "posy", "negy", "posz", "negz"};
    for (int f = 0; f < 6; ++f) {
      io::Image faceImg;
      faceImg.width = faceSize;
      faceImg.height = faceSize;
      faceImg.channels = channels;
      faceImg.data.assign(h_data.begin() + f * faceSize * faceSize * channels,
                          h_data.begin() +
                              (f + 1) * faceSize * faceSize * channels);
      io::save_png(prefix + "_" + faceNames[f] + ".png", faceImg);
    }
  }

  void saveCross(const std::string &filename) const {
    std::vector<float> h_data(faceSize * faceSize * channels * 6);
    data.download(h_data.data(), h_data.size());

    // Layout:
    //      [+Y]
    // [-X] [+Z] [+X] [-Z]
    //      [-Y]
    // Width: 4*size, Height: 3*size
    io::Image crossImg;
    crossImg.width = faceSize * 4;
    crossImg.height = faceSize * 3;
    crossImg.channels = channels;
    crossImg.data.resize(crossImg.width * crossImg.height * channels, 0.0f);

    auto copyFace = [&](int faceIdx, int offsetX, int offsetY) {
      for (int y = 0; y < faceSize; ++y) {
        for (int x = 0; x < faceSize; ++x) {
          for (int c = 0; c < channels; ++c) {
            int srcIdx =
                (faceIdx * faceSize * faceSize + y * faceSize + x) * channels +
                c;
            int dstIdx =
                ((offsetY + y) * crossImg.width + (offsetX + x)) * channels + c;
            crossImg.data[dstIdx] = h_data[srcIdx];
          }
        }
      }
    };

    copyFace(2, faceSize, 0);            // +Y (Top)
    copyFace(1, 0, faceSize);            // -X (Left)
    copyFace(4, faceSize, faceSize);     // +Z (Front)
    copyFace(0, faceSize * 2, faceSize); // +X (Right)
    copyFace(5, faceSize * 3, faceSize); // -Z (Back)
    copyFace(3, faceSize, faceSize * 2); // -Y (Bottom)

    io::save_png(filename, crossImg);
  }
};

} // namespace core
