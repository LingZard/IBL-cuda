#include "io/image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <stdexcept>

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
  // Convert float to byte for PNG
  std::vector<unsigned char> byteData(img.data.size());
  for (size_t i = 0; i < img.data.size(); ++i) {
    // Simple tone mapping / clamping for preview
    float val = img.data[i];
    if (val < 0.0f)
      val = 0.0f;
    if (val > 1.0f)
      val = 1.0f;
    byteData[i] = static_cast<unsigned char>(val * 255.0f);
  }

  if (!stbi_write_png(path.c_str(), img.width, img.height, img.channels,
                      byteData.data(), img.width * img.channels)) {
    std::cerr << "Failed to save PNG image: " << path << std::endl;
  }
}

} // namespace io
