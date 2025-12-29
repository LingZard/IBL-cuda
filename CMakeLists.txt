CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

project(IBL-CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Architecture setting
set(IBL_CUDA_ARCH "89" CACHE STRING "CUDA SM architecture (e.g., 86, 89, 90)")

# Common include directories
set(IBL_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd
)

# Core Library
add_library(ibl_core STATIC
    src/io/image_io.cpp
    src/ibl/equirect_to_cube.cu
    src/ibl/diffuse_irradiance.cu
    src/ibl/specular_prefilter.cu
    src/ibl/brdf_lut.cu
)

target_include_directories(ibl_core PUBLIC 
    ${IBL_INCLUDES}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Main Application
add_executable(ibl_preprocess
    src/app/main.cu
)

target_link_libraries(ibl_preprocess PRIVATE ibl_core)
set_property(TARGET ibl_preprocess PROPERTY CUDA_ARCHITECTURES ${IBL_CUDA_ARCH})
