#include"ntsCudaTest.cuh"
#include <iostream>

__global__ void test_cuda(double* res, int* M, int* N)
{
    double sum = 0.0;
    for(int j =0; j< *M; j++)
    {
        for(int i = 0; i < *N; i++)
        {
          sum = sum + i *(*M);
        }
    }
    *res = sum;
    // std::printf("test sum: %lf\n", *res);
}
void testCuda(double* res, cudaStream_t stream, int M, int N) {
    std::printf("进行了cuda调用\n");
    double* d_A;
    int *d_M;
    int *d_N;
    int nBytes = sizeof(double);
    int mBytes = sizeof(int);
    cudaMallocAsync((double**)&d_A, nBytes, stream);
    cudaMallocAsync((int**)&d_M, mBytes, stream);
    cudaMallocAsync((int**)&d_N, mBytes, stream);
    cudaMemcpyAsync(d_M, &M, mBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_N, &N, mBytes, cudaMemcpyHostToDevice, stream);
    // test_cuda<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(d_A, d_M, d_N);
    cudaMemcpyAsync(res, d_A, nBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::printf("res: %lf\n", *res);
}

void testStream(cudaStream_t stream) {
    int num = 10 * 1024 * 1024 / sizeof(int);
    int * h_a = new int[num];
    int nBytes = num * sizeof(int);
    int * d_a;
    cudaMallocAsync((int **)&d_a, nBytes, stream);
    cudaMemcpyAsync(d_a, h_a, nBytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_a, stream);
}