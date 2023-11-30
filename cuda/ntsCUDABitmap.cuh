//
// Created by toao on 23-2-18.
//

#ifndef GNNMINI_NTSCUDABITMAP_CUH
#define GNNMINI_NTSCUDABITMAP_CUH

#include "ntsCUDA.hpp"
#include "thrust/scan.h"


#define CUDA_WORD_OFFSET(i) ((i) >> 6)
#define CUDA_BIT_OFFSET(i) ((i)&0x3f)


class CudaBitmap{
public:
    size_t size;
    unsigned long long* data;
    cudaStream_t  stream;
    CudaBitmap():size(0), data(nullptr), stream(0) {}
    CudaBitmap(size_t size_) : size(size_), stream(0) {
        cudaMallocAsync((void**)&data, sizeof(unsigned long long) * (CUDA_WORD_OFFSET(size) + 1), stream);
        cudaMemsetAsync(data, 0,  CUDA_WORD_OFFSET(size) * sizeof(unsigned long long), stream);
    }
    CudaBitmap(size_t size_, cudaStream_t stream_): size(size_), stream(stream_) {
        cudaMallocAsync((void**)&data, sizeof(unsigned long long) * (CUDA_WORD_OFFSET(size) + 1), stream);
        cudaMemsetAsync(data, 0,  CUDA_WORD_OFFSET(size) * sizeof(unsigned long long), stream);
    }
    void clear(){
        cudaMemsetAsync(data, 0,  CUDA_WORD_OFFSET(size) * sizeof(unsigned long long), stream);
    }
    void fill() {
        size_t bm_size = CUDA_WORD_OFFSET(size);
        cudaMemset(data, 0xffffffff, CUDA_WORD_OFFSET(size));
        data[bm_size] = 0;
        for (size_t i = (bm_size << 6); i < size; i++) {
            data[bm_size] |= 1ul << CUDA_BIT_OFFSET(i);
        }
    }
    __device__ void set_bit(size_t i) {
        atomicOr(data + CUDA_WORD_OFFSET(i), 1ull << CUDA_BIT_OFFSET(i));
    }
    __device__ unsigned long long get_bit(size_t i) {
        return data[CUDA_WORD_OFFSET(i)] & (1ull << CUDA_BIT_OFFSET(i));
    }
};
__device__ void cuda_set_bit(unsigned long long * data, size_t size, size_t i) {
    atomicOr(data + CUDA_WORD_OFFSET(i), 1ull << CUDA_BIT_OFFSET(i));
}
__device__ unsigned long long cuda_get_bit(unsigned long long * data, size_t size, size_t i) {
    // std::printf("进入了get函数\n");
    // return 1ull;
    return (data[CUDA_WORD_OFFSET(i)] & (1ull << CUDA_BIT_OFFSET(i)));
}
//__global__ void clearCudaBitmap(CudaBitmap* bitmap) {
//    size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
//    size_t bm_size = CUDA_WORD_OFFSET(bitmap->size);
//    for(long i = threadId; i < bm_size; i += blockDim.x*gridDim.x){
//        bitmap->data[i] = 0;
//    }
//}

//__global__ void fillCudaBitmap(CudaBitmap* bitmap) {
//    size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
//    size_t bm_size = CUDA_WORD_OFFSET(bitmap->size);
//    for(long i = threadId; i < bm_size; i += blockDim.x*gridDim.x) {
//        bitmap->data[i] = 0xffffffffffffffff;
//    }
//    if(threadId == 0){
//        bitmap->data[bm_size] = 0;
//        for (size_t i = (bm_size << 6); i < bitmap->size; i++) {
//            bitmap->data[bm_size] |= 1ul << CUDA_BIT_OFFSET(i);
//        }
//    }
//}



#endif //GNNMINI_NTSCUDABITMAP_CUH
