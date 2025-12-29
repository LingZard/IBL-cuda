#include <iostream>
#include <string>
#include "io/image.hpp"
#include "utils/cuda_utils.hpp"

int main(int argc, char** argv) {
    try {
        std::cout << "IBL Preprocessing Tool (CUDA)" << std::endl;
        
        // Basic CUDA check
        int deviceCount = 0;
        CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            std::cerr << "No CUDA devices found!" << std::endl;
            return 1;
        }

        // TODO: Implement actual IBL pipeline here
        std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;
        
        std::string input_path = "assets/golden_gate_hills_4k.hdr";
        std::cout << "Loading: " << input_path << "..." << std::endl;
        
        auto img = io::load_hdr(input_path);
        std::cout << "Loaded: " << img.width << "x" << img.height << " with " << img.channels << " channels." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

