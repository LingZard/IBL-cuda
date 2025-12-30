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

struct CubemapLevel {
  int size;
  CudaBuffer<float> data;
};

struct Cubemap {
  int faceSize;
  int channels;
  int numLevels;
  std::vector<CubemapLevel> levels;

  Cubemap() : faceSize(0), channels(0), numLevels(0) {}

  Cubemap(int size, int comps, int mipLevels = 1)
      : faceSize(size), channels(comps), numLevels(mipLevels) {
    levels.resize(numLevels);
    int currentSize = size;
    for (int i = 0; i < numLevels; ++i) {
      levels[i].size = currentSize;
      levels[i].data.allocate(currentSize * currentSize * comps * 6);
      currentSize /= 2;
      if (currentSize < 1)
        currentSize = 1;
    }
  }

  CudaBuffer<float> &baseLevelData() { return levels[0].data; }
  const CudaBuffer<float> &baseLevelData() const { return levels[0].data; }

  void saveFaces(const std::string &directory, const std::string &prefix,
                 const std::string &ext, int level = 0) const {
    if (level >= numLevels)
      return;
    int size = levels[level].size;
    std::vector<float> h_data(size * size * channels * 6);
    levels[level].data.download(h_data.data(), h_data.size());

    const char *faceNames[] = {"posx", "negx", "posy", "negy", "posz", "negz"};
    for (int f = 0; f < 6; ++f) {
      io::Image faceImg;
      faceImg.width = size;
      faceImg.height = size;
      faceImg.channels = channels;
      faceImg.data.assign(h_data.begin() + f * size * size * channels,
                          h_data.begin() + (f + 1) * size * size * channels);

      std::string path = directory + "/" + prefix + "_L" +
                         std::to_string(level) + "_" + faceNames[f] + ext;
      if (ext == ".png")
        io::save_png(path, faceImg);
      else if (ext == ".hdr")
        io::save_hdr(path, faceImg);
    }
  }

  void saveCross(const std::string &directory, const std::string &filename,
                 const std::string &ext, int level = 0) const {
    if (level >= numLevels)
      return;
    int size = levels[level].size;
    std::vector<float> h_data(size * size * channels * 6);
    levels[level].data.download(h_data.data(), h_data.size());

    // Layout:
    //      [+Y]
    // [-X] [+Z] [+X] [-Z]
    //      [-Y]
    io::Image crossImg;
    crossImg.width = size * 4;
    crossImg.height = size * 3;
    crossImg.channels = channels;
    crossImg.data.resize(crossImg.width * crossImg.height * channels, 0.0f);

    auto copyFace = [&](int faceIdx, int offsetX, int offsetY) {
      for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
          for (int c = 0; c < channels; ++c) {
            int srcIdx = (faceIdx * size * size + y * size + x) * channels + c;
            int dstIdx =
                ((offsetY + y) * crossImg.width + (offsetX + x)) * channels + c;
            crossImg.data[dstIdx] = h_data[srcIdx];
          }
        }
      }
    };

    copyFace(2, size, 0);        // +Y
    copyFace(1, 0, size);        // -X
    copyFace(4, size, size);     // +Z
    copyFace(0, size * 2, size); // +X
    copyFace(5, size * 3, size); // -Z
    copyFace(3, size, size * 2); // -Y

    std::string path = directory + "/" + filename + ext;
    if (ext == ".png")
      io::save_png(path, crossImg);
    else if (ext == ".hdr")
      io::save_hdr(path, crossImg);
  }
};

} // namespace core
