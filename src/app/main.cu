#include "core/cubemap.hpp"
#include "ibl/diffuse_irradiance.hpp"
#include "ibl/equirect_to_cube.hpp"
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
