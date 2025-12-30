#include "core/cubemap.hpp"
#include "ibl/brdf_lut.hpp"
#include "ibl/diffuse_irradiance.hpp"
#include "ibl/equirect_to_cube.hpp"
#include "ibl/specular_prefilter.hpp"
#include "io/image.hpp"
#include "utils/cuda_utils.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void printUsage() {
  std::cout << "Usage: ibl_preprocess <command> <args>" << std::endl;
  std::cout << "Commands:" << std::endl;
  std::cout << "  equirect_to_cube <input.hdr> [faceSize] [output_prefix]"
            << std::endl;
  std::cout << "  diffuse_irradiance <input.hdr> [faceSize] [output_prefix]"
            << std::endl;
  std::cout << "  specular_prefilter <input.hdr> [faceSize] [mipLevels] "
               "[output_prefix]"
            << std::endl;
  std::cout << "  brdf_lut [size] [output_name_no_ext]" << std::endl;
}

// Helper to save a cubemap level to both PNG (Gamma) and HDR
void saveCubemapResults(const core::Cubemap &cubemap, const std::string &prefix,
                        int level = 0) {
  std::cout << "  Saving Level " << level << " to output/png and output/hdr..."
            << std::endl;

  fs::create_directories("output/png");
  fs::create_directories("output/hdr");

  cubemap.saveFaces("output/png", prefix, ".png", level);
  cubemap.saveFaces("output/hdr", prefix, ".hdr", level);
  cubemap.saveCross("output/png",
                    prefix + "_L" + std::to_string(level) + "_cross", ".png",
                    level);
  cubemap.saveCross("output/hdr",
                    prefix + "_L" + std::to_string(level) + "_cross", ".hdr",
                    level);
}

void handleEquirectToCube(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Error: Missing input HDR path for equirect_to_cube"
              << std::endl;
    return;
  }
  std::string inputPath = argv[2];
  int faceSize = (argc > 3) ? std::stoi(argv[3]) : 512;
  std::string outputPrefix = (argc > 4) ? argv[4] : "cubemap";

  std::cout << "--- Equirectangular to Cubemap ---" << std::endl;
  std::cout << "Loading: " << inputPath << "..." << std::endl;
  auto hdrImg = io::load_hdr(inputPath);

  std::cout << "Processing..." << std::endl;
  core::Cubemap cubemap(faceSize, 4);
  ibl::EquirectToCube converter(faceSize);
  converter.process(hdrImg, cubemap);

  saveCubemapResults(cubemap, outputPrefix, 0);
  std::cout << "Done." << std::endl;
}

void handleDiffuseIrradiance(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Error: Missing input HDR path for diffuse_irradiance"
              << std::endl;
    return;
  }
  std::string inputPath = argv[2];
  int faceSize = (argc > 3) ? std::stoi(argv[3]) : 32;
  std::string outputPrefix = (argc > 4) ? argv[4] : "irradiance";

  std::cout << "--- Diffuse Irradiance Generation ---" << std::endl;
  std::cout << "Loading: " << inputPath << "..." << std::endl;
  auto hdrImg = io::load_hdr(inputPath);

  std::cout << "[1/2] Converting to intermediate environment Cubemap..."
            << std::endl;
  int envSize = 512;
  core::Cubemap envCubemap(envSize, 4);
  ibl::EquirectToCube converter(envSize);
  converter.process(hdrImg, envCubemap);

  std::cout << "[2/2] Convolving Irradiance Map (" << faceSize << "x"
            << faceSize << ")..." << std::endl;
  core::Cubemap irradianceMap(faceSize, 4);
  ibl::DiffuseIrradiance irradianceProcessor(faceSize);
  irradianceProcessor.process(envCubemap, irradianceMap);

  saveCubemapResults(irradianceMap, outputPrefix, 0);
  std::cout << "Done." << std::endl;
}

void handleSpecularPrefilter(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Error: Missing input HDR path for specular_prefilter"
              << std::endl;
    return;
  }
  std::string inputPath = argv[2];
  int faceSize = (argc > 3) ? std::stoi(argv[3]) : 128;
  int mipLevels = (argc > 4) ? std::stoi(argv[4]) : 6;
  std::string outputPrefix = (argc > 5) ? argv[5] : "prefilter";

  std::cout << "--- Specular Prefilter Generation ---" << std::endl;
  std::cout << "Loading: " << inputPath << "..." << std::endl;
  auto hdrImg = io::load_hdr(inputPath);

  std::cout << "[1/2] Converting to intermediate environment Cubemap..."
            << std::endl;
  int envSize = 512;
  core::Cubemap envCubemap(envSize, 4);
  ibl::EquirectToCube converter(envSize);
  converter.process(hdrImg, envCubemap);

  std::cout << "[2/2] Filtering " << mipLevels << " Mipmaps..." << std::endl;
  core::Cubemap prefilterMap(faceSize, 4, mipLevels);
  ibl::SpecularPrefilter prefilterProcessor(faceSize, mipLevels);
  prefilterProcessor.process(envCubemap, prefilterMap);

  for (int i = 0; i < mipLevels; ++i) {
    saveCubemapResults(prefilterMap, outputPrefix, i);
  }
  std::cout << "Done." << std::endl;
}

void handleBRDFLUT(int argc, char **argv) {
  int size = (argc > 2) ? std::stoi(argv[2]) : 512;
  std::string outputPrefix = (argc > 3) ? argv[3] : "brdf_lut";

  std::cout << "--- BRDF LUT Generation ---" << std::endl;
  std::cout << "Generating LUT (" << size << "x" << size << ")..." << std::endl;

  io::Image lutImg;
  ibl::BRDFLUT brdfProcessor(size);
  brdfProcessor.process(lutImg);

  std::cout << "Saving results to output/png and output/hdr..." << std::endl;
  fs::create_directories("output/png");
  fs::create_directories("output/hdr");
  io::save_png("output/png/" + outputPrefix + ".png", lutImg);
  io::save_hdr("output/hdr/" + outputPrefix + ".hdr", lutImg);
  std::cout << "Done." << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printUsage();
    return 1;
  }

  try {
    std::string command = argv[1];

    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
      std::cerr << "No CUDA devices found!" << std::endl;
      return 1;
    }

    if (command == "equirect_to_cube") {
      handleEquirectToCube(argc, argv);
    } else if (command == "diffuse_irradiance") {
      handleDiffuseIrradiance(argc, argv);
    } else if (command == "specular_prefilter") {
      handleSpecularPrefilter(argc, argv);
    } else if (command == "brdf_lut") {
      handleBRDFLUT(argc, argv);
    } else {
      std::cerr << "Unknown command: " << command << std::endl;
      printUsage();
      return 1;
    }

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
