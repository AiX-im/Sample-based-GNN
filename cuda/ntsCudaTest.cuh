#include <cuda_runtime.h>
#include "cuda_type.h"
void testCuda(double* res, cudaStream_t stream, int M = 10000, int N = 30000000);
void testStream(cudaStream_t stream);