#pragma once

#include <string>
#include <vector>

namespace io {

struct Image {
  int width;
  int height;
  int channels;
  std::vector<float> data; // Host data
};

Image load_hdr(const std::string &path);
void save_hdr(const std::string &path, const Image &img);
void save_png(const std::string &path, const Image &img);

} // namespace io
