#include "io/image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace io {

Image load_hdr(const std::string &path) {
  int width, height, nrComponents;
  float *data = stbi_loadf(path.c_str(), &width, &height, &nrComponents, 0);

  if (!data) {
    throw std::runtime_error("Failed to load HDR image: " + path);
  }

  Image img;
  img.width = width;
  img.height = height;
  img.channels = nrComponents;
  img.data.assign(data, data + (width * height * nrComponents));

  stbi_image_free(data);
  return img;
}

void save_hdr(const std::string &path, const Image &img) {
  if (!stbi_write_hdr(path.c_str(), img.width, img.height, img.channels,
                      img.data.data())) {
    std::cerr << "Failed to save HDR image: " << path << std::endl;
  }
}

void save_png(const std::string &path, const Image &img) {
  // Convert float to byte for PNG with Gamma Correction (2.2)
  // Gamma is only applied to RGB channels. Alpha is preserved.
  std::vector<unsigned char> byteData(img.data.size());
  const int channels = img.channels;
  const size_t pixelCount = img.data.size() / (size_t)channels;
  const float invGamma = 1.0f / 2.2f;

  for (size_t p = 0; p < pixelCount; ++p) {
    for (int c = 0; c < channels; ++c) {
      float val = img.data[p * channels + c];
      if (c < 3) { // RGB
        val = (val < 0.0f) ? 0.0f : ((val > 1.0f) ? 1.0f : val);
        val = powf(val, invGamma);
      } else { // Alpha
        // Keep as is, or clamp if necessary
        val = (val < 0.0f) ? 0.0f : ((val > 1.0f) ? 1.0f : val);
      }
      byteData[p * channels + c] = static_cast<unsigned char>(val * 255.0f);
    }
  }

  if (!stbi_write_png(path.c_str(), img.width, img.height, img.channels,
                      byteData.data(), img.width * img.channels)) {
    std::cerr << "Failed to save PNG image: " << path << std::endl;
  }
}

} // namespace io
