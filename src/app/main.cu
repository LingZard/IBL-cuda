#include "core/cubemap.hpp"
#include "ibl/brdf_lut.hpp"
#include "ibl/diffuse_irradiance.hpp"
#include "ibl/equirect_to_cube.hpp"
#include "ibl/specular_prefilter.hpp"
#include "io/image.hpp"
#include "utils/cuda_utils.hpp"
#include <iostream>
#include <string>
#include <vector>

void printUsage() {
  std::cout << "Usage: ibl_preprocess <command> <args>" << std::endl;
  std::cout << "Commands:" << std::endl;
  std::cout << "  equirect_to_cube <input.hdr> [faceSize] [output_prefix]"
            << std::endl;
  std::cout << "  diffuse_irradiance <input.hdr> [faceSize] [output_prefix]"
            << std::endl;
  std::cout << "  specular_prefilter <input.hdr> [faceSize] [output_prefix]"
            << std::endl;
  std::cout << "  brdf_lut [size] [output_name.png]" << std::endl;
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

  std::cout << "Saving results to " << outputPrefix << "_*.png" << std::endl;
  cubemap.saveFaces(outputPrefix);
  cubemap.saveCross(outputPrefix + "_cross.png");
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

  // 1. Convert to environment Cubemap first (high res)
  std::cout << "[1/2] Converting to intermediate environment Cubemap..."
            << std::endl;
  int envSize = 512;
  core::Cubemap envCubemap(envSize, 4);
  ibl::EquirectToCube converter(envSize);
  converter.process(hdrImg, envCubemap);

  // 2. Generate Irradiance Map
  std::cout << "[2/2] Convolving Irradiance Map (" << faceSize << "x"
            << faceSize << ")..." << std::endl;
  core::Cubemap irradianceMap(faceSize, 4);
  ibl::DiffuseIrradiance irradianceProcessor(faceSize);
  irradianceProcessor.process(envCubemap, irradianceMap);

  std::cout << "Saving results to " << outputPrefix << "_*.png" << std::endl;
  irradianceMap.saveFaces(outputPrefix);
  irradianceMap.saveCross(outputPrefix + "_cross.png");
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
  std::string outputPrefix = (argc > 4) ? argv[4] : "prefilter";

  std::cout << "--- Specular Prefilter Generation ---" << std::endl;
  std::cout << "Loading: " << inputPath << "..." << std::endl;
  auto hdrImg = io::load_hdr(inputPath);

  std::cout << "[1/2] Converting to intermediate environment Cubemap..."
            << std::endl;
  int envSize = 512;
  core::Cubemap envCubemap(envSize, 4);
  ibl::EquirectToCube converter(envSize);
  converter.process(hdrImg, envCubemap);

  std::cout << "[2/2] Filtering Mipmaps..." << std::endl;
  int mipLevels = 5;
  core::Cubemap prefilterMap(faceSize, 4, mipLevels);
  ibl::SpecularPrefilter prefilterProcessor(faceSize, mipLevels);
  prefilterProcessor.process(envCubemap, prefilterMap);

  std::cout << "Saving results (all levels)..." << std::endl;
  for (int i = 0; i < mipLevels; ++i) {
    prefilterMap.saveFaces(outputPrefix, i);
    prefilterMap.saveCross(
        outputPrefix + "_L" + std::to_string(i) + "_cross.png", i);
  }
  std::cout << "Done." << std::endl;
}

void handleBRDFLUT(int argc, char **argv) {
  int size = (argc > 2) ? std::stoi(argv[2]) : 512;
  std::string outputName = (argc > 3) ? argv[3] : "brdf_lut.png";

  std::cout << "--- BRDF LUT Generation ---" << std::endl;
  std::cout << "Generating LUT (" << size << "x" << size << ")..." << std::endl;

  io::Image lutImg;
  ibl::BRDFLUT brdfProcessor(size);
  brdfProcessor.process(lutImg);

  std::cout << "Saving results to " << outputName << std::endl;
  io::save_png(outputName, lutImg);
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
