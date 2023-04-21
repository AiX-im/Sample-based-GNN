project(cuda_propagate)
# cmake_minimum_required(VERSION 2.8)
cmake_minimum_required(VERSION 3.14)
find_package(CUDA REQUIRED)
 
file(GLOB_RECURSE CURRENT_HEADERS &.hpp *.cuh *.h)
file(GLOB CURRENT_SOURCES *.cu *.cpp)
 
# set(CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75;-O3;-G;-g)

list(APPEND CUDA_NVCC_FLAGS "-std=c++17")
 
source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})
 
cuda_add_library(cuda_propagate STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
 
#not used in this demo
