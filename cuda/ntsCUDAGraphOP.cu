
#include <random>
#include"cuda_type.h"
#include "ntsCUDA.hpp"
#include <iostream>
#include <boost/stacktrace.hpp>


#if CUDA_ENABLE
#include "ntsCUDAFuseKernel.cuh"
#include "ntsCUDADistKernel.cuh"
#include "ntsCUDATransferKernel.cuh"
#include "thrust/scan.h"
#include "thrust/device_ptr.h"
#include <thread>


#endif

#if CUDA_ENABLE
#define CHECK_CUDA_RESULT(N) {											\
	cudaError_t result = N;												\
	if (result != 0) {													\
		printf("thread 0x%lx CUDA call on file %s line %d returned code %d, error: %s\n", \
        std::this_thread::get_id(), __FILE__, __LINE__, result, cudaGetErrorString(result));       \
        std::cout << boost::stacktrace::stacktrace();                                 \
		exit(1);														\
	} }
#endif

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at file %s line %d with error: %s (%d)\n",         \
               __FILE__, __LINE__, cusparseGetErrorString(status), status);    \
        std::cout << boost::stacktrace::stacktrace();                             \
       exit(EXIT_FAILURE);                                                     \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                   \
{                                                                              \
    cublasStatus_t status = (func);                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
        printf("CUBLAS API failed at file %s line %d with error: %d\n",         \
               __FILE__, __LINE__, status);                                  \
        std::cout << boost::stacktrace::stacktrace();                       \
       exit(EXIT_FAILURE);                                                     \
    }                                                                          \
}

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// TODO: Toao debug

uint64_t Cuda_Stream::total_cache_hit = 0;
uint64_t Cuda_Stream::total_sample_num = 0;
uint64_t Cuda_Stream::total_transfer_node = 0;

template<typename T>
T get_cuda_array_num(T* arr, int index) {
    T value;
    cudaMemcpy(&value, arr+index, sizeof(T), cudaMemcpyDeviceToHost);
    return value;
}


__global__ void print_cache_num(VertexId_CUDA *destination_vertex, VertexId_CUDA *dev_cachemap, VertexId_CUDA vertex_size) {
    uint32_t sum = 0;
    for(VertexId_CUDA i = 0; i < vertex_size; i++) {
        if(dev_cachemap[destination_vertex[i]] != -1){
            sum++;
        }
    }
    std::printf("顶点总数: %u, 缓存数量: %u\n", vertex_size, sum);
}

void* getDevicePointer(void* host_data_to_device){
#if CUDA_ENABLE
    void* dev_host_data_to_device;
    CHECK_CUDA_RESULT(cudaHostGetDevicePointer(&dev_host_data_to_device,host_data_to_device,0));
    return dev_host_data_to_device;
#else
    printf("CUDA DISABLED getDevicePointer\n");
    exit(0);   
#endif 

}

void* cudaMallocPinned(long size_of_bytes){

#if CUDA_ENABLE       
    void *data=NULL;
   CHECK_CUDA_RESULT(cudaHostAlloc(&data,size_of_bytes, cudaHostAllocMapped));
    return data;
#else
    printf("CUDA DISABLED cudaMallocPinned\n");
    exit(0);   
#endif
}

void * cudaMallocZero(long size_of_bytes){

#if CUDA_ENABLE
    void *data=NULL;
    CHECK_CUDA_RESULT(cudaMallocManaged((void**)&data,size_of_bytes));
    return data;
#else
    printf("CUDA DISABLED cudaMallocZero\n");
    exit(0);
#endif
}


void* cudaMallocPinnedMulti(long size_of_bytes){

#if CUDA_ENABLE
    void *data=NULL;
    CHECK_CUDA_RESULT(cudaHostAlloc(&data,size_of_bytes, cudaHostAllocPortable));
    return data;
#else
    printf("CUDA DISABLED cudaMallocPinnedMulti\n");
    exit(0);
#endif
}

void cudaSetMemAsync(void* mem, int value, size_t size, cudaStream_t stream){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemsetAsync(mem, value, size, stream));
#else
    printf("CUDA DISABLED cudaSetMemAsync\n");
    exit(0);
#endif

}

void cudaSetUsingDevice(int device_id){
#if CUDA_ENABLE
//    if(device_id != 0) {
//        std::cout << boost::stacktrace::stacktrace() << std::endl;
//    }
//    assert(device_id == 0);
    CHECK_CUDA_RESULT(cudaSetDevice(device_id));
//       printf("malloc finished\n");
#else
    printf("CUDA DISABLED cudaSetDevice\n");
       exit(0);
#endif
}

void* cudaMallocGPU(long size_of_bytes){
#if CUDA_ENABLE
       void *data=NULL;
       CHECK_CUDA_RESULT(cudaMalloc(&data,size_of_bytes));
//       printf("malloc finished\n");
       return data;
#else
       printf("CUDA DISABLED cudaMallocGPU\n");
       exit(0);   
#endif  
}

//ncclComm_t* NCCL_Communicator::ncclComms = nullptr;

void initNCCLComm(ncclComm_t* comms, int nDev, int* devs) {
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
}

void destroyNCCLComm(ncclComm_t comm) {
    NCCLCHECK(ncclCommDestroy(comm));
}
void allReduceNCCL(void* send_buffer, void* recv_buffer, size_t element_num, ncclComm_t comm,
                   cudaStream_t cudaStream, int device_id) {
//    std::printf("allReduce device id: %d\n", device_id);
    CHECK_CUDA_RESULT(cudaSetDevice(device_id));
    NCCLCHECK(ncclAllReduce(send_buffer, recv_buffer, element_num, ncclFloat,
                            ncclSum, comm, cudaStream));
}

void broadcastNCCL(void* send_buffer, size_t element_num, ncclComm_t comm, cudaStream_t cudaStream,
                   int device_id) {
    CHECK_CUDA_RESULT(cudaSetDevice(device_id));
    std::printf("device id: %d\n", device_id);
    NCCLCHECK(ncclBcast(send_buffer, element_num, ncclFloat, device_id, comm,  cudaStream));
}

void allGatherNCCL(void*send_buffer, void* recv_buffer, size_t element_num, ncclComm_t comm,
                   cudaStream_t cudaStream, int device_id) {
    CHECK_CUDA_RESULT(cudaSetDevice(device_id));
//    CHECK_CUDA_RESULT(cudaStreamSynchronize(cudaStream));
    NCCLCHECK(ncclAllGather(send_buffer, recv_buffer, element_num, ncclFloat, comm, cudaStream));
}



Cuda_Stream::Cuda_Stream(){
#if CUDA_ENABLE
       CHECK_CUDA_RESULT(cudaStreamCreate(&stream));
       CHECK_CUSPARSE( cusparseCreate(&sparse_handle) );
       CHECK_CUSPARSE(cusparseSetStream(sparse_handle, stream));
       CHECK_CUBLAS(cublasCreate(&blas_handle));
       CHECK_CUBLAS(cublasSetStream(blas_handle, stream));
#else
       printf("CUDA DISABLED Cuda_Stream::Cuda_Stream\n");
       exit(0);  
#endif  
}

void Cuda_Stream::destory_Stream(){
#if CUDA_ENABLE
    CHECK_CUSPARSE(cusparseDestroy(sparse_handle));
    CHECK_CUBLAS(cublasDestroy(blas_handle));
    CHECK_CUDA_RESULT(cudaStreamDestroy(stream));

#else
       printf("CUDA DISABLED Cuda_Stream::Cuda_Stream\n");
       exit(0);   
#endif     

}
inline cudaStream_t Cuda_Stream::getStream(){
    
#if CUDA_ENABLE
        return stream;
#else
       printf("CUDA DISABLED Cuda_Stream::getStream\n");
       exit(0);   
#endif   
}

void Cuda_Stream::setNewStream(cudaStream_t cudaStream) {
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaStreamDestroy(stream));
    this->stream = cudaStream;
    CHECK_CUSPARSE(cusparseSetStream(sparse_handle, stream));
       CHECK_CUBLAS(cublasSetStream(blas_handle, stream));

#else
    printf("CUDA DISABLED Cuda_Stream::getStream\n");
    exit(0);
#endif
}

void ResetDevice(){
#if CUDA_ENABLE
   cudaDeviceReset();
#else
       printf("CUDA DISABLED ResetDevice\n");
       exit(0);   
#endif   
 
}
void Cuda_Stream::CUDA_DEVICE_SYNCHRONIZE(){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaStreamSynchronize(stream));
#else
       printf("CUDA DISABLED Cuda_Stream::CUDA_DEVICE_SYNCHRONIZE\n");
       exit(0);   
#endif   
}

void Cuda_Stream::CUDA_SYNCHRONIZE_ALL(){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaDeviceSynchronize());
#else
    printf("CUDA DISABLED Cuda_Stream::CUDA_DEVICE_SYNCHRONIZE\n");
       exit(0);
#endif
}

void Cuda_Stream::move_bytes_out(VertexId_CUDA* h_pointer, VertexId_CUDA* d_pointer, int size){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpyAsync(h_pointer, d_pointer, size * (sizeof(VertexId_CUDA)), cudaMemcpyDeviceToHost, stream));
#else
       printf("CUDA DISABLED Cuda_Stream::move_result_out\n");
       exit(0);   
#endif   
}


void Cuda_Stream::move_result_out(float* output,float* input, VertexId_CUDA src,VertexId_CUDA dst, int feature_size,bool sync){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpyAsync(output,input,((long)(dst-src))*feature_size*(sizeof(int)), cudaMemcpyDeviceToHost,stream));
#else
       printf("CUDA DISABLED Cuda_Stream::move_result_out\n");
       exit(0);   
#endif   
}
void Cuda_Stream::move_data_in(float* d_pointer,float* h_pointer, VertexId_CUDA start, VertexId_CUDA end, int feature_size,bool sync){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpyAsync(d_pointer,h_pointer,((long)(end-start))*feature_size*(sizeof(float)), cudaMemcpyHostToDevice,stream));
#else
       printf("CUDA DISABLED Cuda_Stream::move_data_in\n");
       exit(0);   
#endif   
  
}
void Cuda_Stream::move_edge_in(VertexId_CUDA* d_pointer,VertexId_CUDA* h_pointer, VertexId_CUDA start, VertexId_CUDA end, int feature_size,bool sync){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpyAsync(d_pointer,h_pointer,((long)(end-start))*feature_size*(sizeof(VertexId_CUDA)), cudaMemcpyHostToDevice,stream));
#else
       printf("CUDA DISABLED Cuda_Stream::move_edge_in\n");
       exit(0);   
#endif       
}
void Cuda_Stream::aggregate_comm_result(float* aggregate_buffer,float *input_buffer,VertexId_CUDA data_size,int feature_size,int partition_offset, bool sync){
#if CUDA_ENABLE
    aggregate_data_buffer<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(aggregate_buffer,input_buffer,data_size,feature_size,partition_offset,sync);
#else
       printf("CUDA DISABLED Cuda_Stream::aggregate_comm_result\n");
       exit(0);   
#endif     
}

void Cuda_Stream::aggregate_comm_result_debug(float* aggregate_buffer,float *input_buffer,VertexId_CUDA data_size,VertexId_CUDA feature_size,VertexId_CUDA partition_start,VertexId_CUDA partition_end, bool sync){
#if CUDA_ENABLE
    aggregate_data_buffer_debug<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(aggregate_buffer,input_buffer,data_size,feature_size,partition_start,partition_end,sync);
#else
       printf("CUDA DISABLED Cuda_Stream::aggregate_comm_result_debug\n");
       exit(0);   
#endif 
}

void Cuda_Stream::deSerializeToGPU(float* input_gpu_buffer,float *input_buffer,VertexId_CUDA data_size,VertexId_CUDA feature_size,VertexId_CUDA partition_start,VertexId_CUDA partition_end, bool sync){
#if CUDA_ENABLE
    deSerializeToGPUkernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(input_gpu_buffer,input_buffer,data_size,feature_size,partition_start,partition_end,sync);
#else
       printf("CUDA DISABLED Cuda_Stream::deSerializeToGPU\n");
       exit(0);   
#endif  
}
void Cuda_Stream::Gather_By_Dst_From_Src(float* input,float* output,float* weight_forward,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,//graph
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight,bool tensor_weight){
#if CUDA_ENABLE
       if(with_weight){
           if(tensor_weight){
    //		aggregate_kernel_from_src_tensor_weight<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE,0,stream>>>(
    //			row_indices, column_offset, input, output, weight_forward,
    //				src_start, dst_start, batch_size, feature_size);
                  printf("aggregate_kernel_from_src_tensor_weight_optim_nts");
           }else{
                  aggregate_kernel_from_src_with_weight<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
                  row_indices, column_offset, input, output, weight_forward,
                         src_start, dst_start, batch_size, feature_size);
           }
       }else{
              aggregate_kernel_from_src_without_weight<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
                     row_indices, column_offset, input, output, weight_forward, 
                            src_start, dst_start, batch_size, feature_size);
       }




       
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Src\n");
       exit(0);   
#endif  
        
}


template<typename T>
void print_cuda_sum(T* data, size_t len, char* msg, int pIndex=-1) {
    T* tmp_result = new T[len];

    cudaMemcpy(tmp_result, data, len * sizeof(T), cudaMemcpyDeviceToHost);
    double sum = 0.0;
    T max = std::numeric_limits<T>::min();
    for(long i = 0; i < len; i++) {
        sum += tmp_result[i];
        max = std::max(max, tmp_result[i]);
    }
    if(pIndex == -1) {
        pIndex = len - 1;
    }
    std::cout << msg << "第" << pIndex << "个元素: " << tmp_result[pIndex] << ", 最大值为："
              << max << ", 总和: " << sum << ", 均值: " << sum / len << std::endl;
    // std::printf("%s最后一个元素：结果总和：%lf\n", msg, sum);
    std::printf("\tthread 0x%lx gpu avg: %.4lf\n", std::this_thread::get_id(), sum/len);
    delete []tmp_result;
}


template<typename T>
void print_cpu_sum(T* tmp_result, size_t len, char* msg, int pIndex=-1) {
//    T* tmp_result = new T[len];

//    cudaMemcpy(tmp_result, data, len * sizeof(T), cudaMemcpyDeviceToHost);
    double sum = 0.0;
    T max = std::numeric_limits<T>::min();
    for(long i = 0; i < len; i++) {
        sum += tmp_result[i];
        max = std::max(max, tmp_result[i]);
    }
    if(pIndex == -1) {
        pIndex = len - 1;
    }
    std::cout << msg << "第" << pIndex << "个元素: " << tmp_result[pIndex] << ", 最大值为："
              << max << ", 总和: " << sum << ", 均值: " << sum / len << std::endl;
    std::printf("\tthread 0x%lx cpu avg: %.4lf\n", std::this_thread::get_id(), sum/len);
    // std::printf("%s最后一个元素：结果总和：%lf\n", msg, sum);
//    delete []tmp_result;
}

template<typename T>
void single_print_cuda_sum(T* data, size_t len, char* msg, int pIndex) {
    print_cuda_sum(data, len, msg, pIndex);
}


void Cuda_Stream::Gather_By_Dst_From_Src_Spmm(float* input,float* output,float* weight_forward,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset, VertexId_CUDA column_num, //graph
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight,bool tensor_weight){
#if CUDA_ENABLE
//    std::printf("real edges: %d, record edges: %d\n", get_cuda_array_num(column_offset, batch_size), edges);
//         if(with_weight){
//             if(tensor_weight){
// //		aggregate_kernel_from_src_tensor_weight<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE,0,stream>>>(
// //			row_indices, column_offset, input, output, weight_forward, 
// //				src_start, dst_start, batch_size, feature_size);
//                 printf("aggregate_kernel_from_src_tensor_weight_optim_nts");
//             }else{
//                 aggregate_kernel_from_src_with_weight<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
// 			row_indices, column_offset, input, output, weight_forward, 
// 				src_start, dst_start, batch_size, feature_size);
//             }
//         }
//         else{
//                 aggregate_kernel_from_src_without_weight<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
//                         row_indices, column_offset, input, output, weight_forward, 
//                                 src_start, dst_start, batch_size, feature_size);
//         }
       // 检查并处理buffer空间不足的问题
       auto total_size = 0;
       if(!with_weight) {
              total_size += edges*sizeof(float);
       }
       if(cuda_buffer_size < total_size){
              auto new_size = total_size * 2;
              if(cuda_buffer_size != 0) {
                     cudaFree(cuda_buffer);
              }
              cudaMalloc((void**)&cuda_buffer, new_size);
              cuda_buffer_size = new_size;
       }

       // 为需要的空间分配buffer
       size_t buffer_cur = 0;
       if(!with_weight){
              weight_forward = (float*)cuda_buffer;
              buffer_cur += edges*sizeof(float);
       }
       // float* input_tran = (float*)(cuda_buffer + buffer_cur);
       // buffer_cur += feature_size * column_num * sizeof(float);
       // float* output_tran = (float*)(cuda_buffer + buffer_cur);
       // buffer_cur += batch_size*feature_size*sizeof(float);

       assert(src_start == 0);
       assert(sparse_handle != NULL);
       cusparseSpMatDescr_t matA;
       cusparseDnMatDescr_t matB, matC;
       void* dBuffer = NULL;
       size_t bufferSize = 0;
       float alpha = 1.0f;
       float beta  = 0.0f;
       // std::printf("edges: %d\n", edges);
       // std::printf("row: %d, column: %d\n", batch_size, column_num);
       if(!with_weight){
              // CHECK_CUDA_RESULT(cudaMallocAsync(&weight_forward, edges*sizeof(float), stream));
           // 0x0000803F 即1.0
           CHECK_CUDA_RESULT(cudaMemsetAsync(&weight_forward, 0x0000803F, sizeof(float) * edges, stream));
       }
//     cusparseHandle_t     sparse_handle = NULL;
//     CHECK_CUSPARSE( cusparseCreate(&sparse_handle) )
//     cusparseSetStream(sparse_handle, 0);

       // cudaStreamSynchronize(stream);
       // std::printf("column num: %d\n", column_num);
       // print_cuda_sum(row_indices, edges, "row_indices");
       // print_cuda_sum(column_offset, batch_size+1, "column offset");
       // print_cuda_sum(column_offset, batch_size+1, "column offset", 0);
       // print_cuda_sum(column_offset, batch_size+1, "column offset", batch_size);
       // print_cuda_sum(weight_forward, edges, "weight_forward");
       // print_cuda_sum(input, column_num*feature_size, "input");
       // print_cuda_sum(output, batch_size * feature_size, "output");
       // std::printf("CUDA v%d.%d\n", CUDART_VERSION/1000, CUDART_VERSION/10%100);
//       CHECK_CUSPARSE(cusparseCreateCsr(&matA, batch_size, column_num, edges, column_offset, row_indices, weight_forward,
//                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsc(&matA, column_num, batch_size, edges, column_offset, row_indices, weight_forward,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
       // float* input_tran;
       //  float *B = NULL;
       //  CHECK_CUDA_RESULT(cudaMalloc((void**)&input_tran, feature_size * column_num * sizeof(float)));
       // CHECK_CUBLAS(cublasSgeam( blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, column_num, feature_size, &alpha, input, feature_size, &beta, input, column_num, input_tran, column_num));
       // float* output_tran;
       // cudaMallocAsync((void**)&output_tran, batch_size*feature_size*sizeof(float), stream);
       // cudaMalloc((void**)&input_tran, column_num*feature_size*sizeof(float));
       // cudaMalloc((void**)&output_tran, batch_size*feature_size*sizeof(float));
       
       // print_cuda_sum(input_tran, column_num*feature_size, "input_tran");

       CHECK_CUSPARSE(cusparseCreateDnMat(&matB, column_num, feature_size, feature_size, input, CUDA_R_32F, CUSPARSE_ORDER_ROW));
       // CHECK_CUSPARSE(cusparseCreateDnMat(&matB, column_num, feature_size, feature_size, input, CUDA_R_32F, CUSPARSE_ORDER_ROW));
       CHECK_CUSPARSE(cusparseCreateDnMat(&matC, batch_size, feature_size, feature_size, output, CUDA_R_32F, CUSPARSE_ORDER_ROW));
       CHECK_CUSPARSE(cusparseSpMM_bufferSize(
              sparse_handle,
              CUSPARSE_OPERATION_TRANSPOSE,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              &alpha, matA, matB, &beta, matC, CUDA_R_32F,
              CUSPARSE_SPMM_CSR_ALG2, &bufferSize
       ));
       total_size += bufferSize;
       if(buffer_cur + bufferSize < cuda_buffer_size) {
              dBuffer = cuda_buffer + buffer_cur;
              buffer_cur += bufferSize;
       } else {
              CHECK_CUDA_RESULT(cudaMallocAsync(&dBuffer, bufferSize, stream));
       }


       CHECK_CUSPARSE(cusparseSpMM(
              sparse_handle,
              CUSPARSE_OPERATION_TRANSPOSE,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              &alpha, matA, matB, &beta, matC, CUDA_R_32F,
              CUSPARSE_SPMM_CSR_ALG2, dBuffer
       ));
       // 矩阵行列转换
       // CHECK_CUBLAS(cublasSgeam( blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, feature_size, batch_size, &alpha, output_tran, batch_size, &beta, output_tran, feature_size, output, feature_size));
       

       CHECK_CUSPARSE(cusparseDestroySpMat(matA));
       CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
       CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
       //     CHECK_CUSPARSE( cusparseDestroy(sparse_handle) )

       // cudaFreeAsync(dBuffer, stream);
       // cudaFreeAsync(input_tran, stream);
       // cudaFreeAsync(output_tran, stream);
       // if(!with_weight){
       //        cudaFreeAsync(weight_forward, stream);
       // }
       
       if(cuda_buffer_size < total_size){
              auto new_size = total_size * 2;
              cudaFreeAsync(dBuffer, stream);
              if(cuda_buffer_size != 0) {
                     cudaFree(cuda_buffer);
              }
              cudaMalloc((void**)&cuda_buffer, new_size);
              cuda_buffer_size = new_size;
       }

       // std::printf("edges: %d\n", edges);
       // print_cuda_sum(weight_forward, edges, "After Matrix weight_forward");

       // cudaStreamSynchronize(stream);
       // print_cuda_sum(weight_forward, edges, "weight_forward");
       // print_cuda_sum(input_tran, column_num* feature_size, "input_tran");
       // print_cuda_sum(output_tran, batch_size * feature_size, "result");
       
       // print_cuda_sum(output, batch_size * feature_size, "destroyed result");

       // cudaStreamSynchronize(stream);
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Src\n");
       exit(0);   
#endif  
        
}


void Cuda_Stream::Push_From_Dst_To_Src(float* input,float* output,float* weight_forward,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,//graph
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight,bool tensor_weight){
#if CUDA_ENABLE
        if(with_weight){
            if(tensor_weight){
//		aggregate_kernel_from_src_tensor_weight<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE,0,stream>>>(
//			row_indices, column_offset, input, output, weight_forward, 
//				src_start, dst_start, batch_size, feature_size);
                printf("aggregate_kernel_from_src_tensor_weight_optim_nts");
            }else{
                push_kernel_from_dst_with_weight<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_indices, column_offset, input, output, weight_forward, 
				src_start, dst_start, batch_size, feature_size);
            }
        }
        else{
                aggregate_kernel_from_src_without_weight<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
                        row_indices, column_offset, input, output, weight_forward, 
                                src_start, dst_start, batch_size, feature_size);
        }
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Src\n");
       exit(0);   
#endif  
        
}

void Cuda_Stream::Push_From_Dst_To_Src_Spmm(float* input,float* output,float* weight_forward,//data
                                       VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,int column_num,//graph
                                       VertexId_CUDA src_start, VertexId_CUDA src_end,
                                       VertexId_CUDA dst_start, VertexId_CUDA dst_end,
                                       VertexId_CUDA edges,VertexId_CUDA batch_size,
                                       VertexId_CUDA feature_size,bool with_weight,bool tensor_weight){
#if CUDA_ENABLE
    auto total_size = 0;
//    edges = get_cuda_array_num(column_offset, batch_size);
    if(!with_weight) {
        total_size += edges*sizeof(float);
    }
    if(cuda_buffer_size < total_size){
        auto new_size = total_size * 2;
        if(cuda_buffer_size != 0) {
            cudaFree(cuda_buffer);
        }
        cudaMalloc((void**)&cuda_buffer, new_size);
        cuda_buffer_size = new_size;
    }

    // 为需要的空间分配buffer
    size_t buffer_cur = 0;
    if(!with_weight){
        weight_forward = (float*)cuda_buffer;
        buffer_cur += edges*sizeof(float);
    }
    // float* input_tran = (float*)(cuda_buffer + buffer_cur);
    // buffer_cur += feature_size * column_num * sizeof(float);
    // float* output_tran = (float*)(cuda_buffer + buffer_cur);
    // buffer_cur += batch_size*feature_size*sizeof(float);

    assert(src_start == 0);
    assert(sparse_handle != NULL);
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    float alpha = 1.0f;
    float beta  = 0.0f;
    // std::printf("edges: %d\n", edges);
    // std::printf("row: %d, column: %d\n", batch_size, column_num);
    if(!with_weight){
        // CHECK_CUDA_RESULT(cudaMallocAsync(&weight_forward, edges*sizeof(float), stream));
        // 0x0000803F 即1.0
        CHECK_CUDA_RESULT(cudaMemsetAsync(&weight_forward, 0x0000803F, sizeof(float) * edges, stream));
    }
//     cusparseHandle_t     sparse_handle = NULL;
//     CHECK_CUSPARSE( cusparseCreate(&sparse_handle) )
//     cusparseSetStream(sparse_handle, 0);

    // cudaStreamSynchronize(stream);
    // std::printf("column num: %d\n", column_num);
    // print_cuda_sum(row_indices, edges, "row_indices");
    // print_cuda_sum(column_offset, batch_size+1, "column offset");
    // print_cuda_sum(column_offset, batch_size+1, "column offset", 0);
    // print_cuda_sum(column_offset, batch_size+1, "column offset", batch_size);
    // print_cuda_sum(weight_forward, edges, "weight_forward");
    // print_cuda_sum(input, column_num*feature_size, "input");
    // print_cuda_sum(output, batch_size * feature_size, "output");
    // std::printf("CUDA v%d.%d\n", CUDART_VERSION/1000, CUDART_VERSION/10%100);
//    std::printf("column num: %d, batch size: %d, feature size: %d, edges: %d\n", column_num, batch_size, feature_size, edges);
//    print_cuda_sum(column_offset, batch_size + 1, "column_offset");
//    print_cuda_sum(row_indices,edges, "row_indices");
//    CHECK_CUSPARSE(cusparseCreateCsr(&matA, batch_size, column_num, edges, column_offset, row_indices, weight_forward,
//                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseCreateCsc(&matA, column_num, batch_size, edges, column_offset, row_indices, weight_forward,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // float* input_tran;
    //  float *B = NULL;
    //  CHECK_CUDA_RESULT(cudaMalloc((void**)&input_tran, feature_size * column_num * sizeof(float)));
    // CHECK_CUBLAS(cublasSgeam( blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, column_num, feature_size, &alpha, input, feature_size, &beta, input, column_num, input_tran, column_num));
    // float* output_tran;
    // cudaMallocAsync((void**)&output_tran, batch_size*feature_size*sizeof(float), stream);
    // cudaMalloc((void**)&input_tran, column_num*feature_size*sizeof(float));
    // cudaMalloc((void**)&output_tran, batch_size*feature_size*sizeof(float));

    // print_cuda_sum(input_tran, column_num*feature_size, "input_tran");

    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, batch_size, feature_size, feature_size, input, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // CHECK_CUSPARSE(cusparseCreateDnMat(&matB, column_num, feature_size, feature_size, input, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, column_num, feature_size, feature_size, output, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            sparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_CSR_ALG2, &bufferSize
    ));
    total_size += bufferSize;
    if(buffer_cur + bufferSize < cuda_buffer_size) {
        dBuffer = cuda_buffer + buffer_cur;
        buffer_cur += bufferSize;
    } else {
        CHECK_CUDA_RESULT(cudaMallocAsync(&dBuffer, bufferSize, stream));
//        CHECK_CUDA_RESULT(cudaMalloc(&dBuffer, bufferSize));
    }


    CHECK_CUSPARSE(cusparseSpMM(
            sparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_CSR_ALG2, dBuffer
    ));
    // 矩阵行列转换
    // CHECK_CUBLAS(cublasSgeam( blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, feature_size, batch_size, &alpha, output_tran, batch_size, &beta, output_tran, feature_size, output, feature_size));


    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    //     CHECK_CUSPARSE( cusparseDestroy(sparse_handle) )

    // cudaFreeAsync(dBuffer, stream);
    // cudaFreeAsync(input_tran, stream);
    // cudaFreeAsync(output_tran, stream);
    // if(!with_weight){
    //        cudaFreeAsync(weight_forward, stream);
    // }

    if(cuda_buffer_size < total_size){
        auto new_size = total_size * 2;
        cudaFreeAsync(dBuffer, stream);
        if(cuda_buffer_size != 0) {
            cudaFree(cuda_buffer);
        }
        cudaMalloc((void**)&cuda_buffer, new_size);
        cuda_buffer_size = new_size;
    }

    // std::printf("edges: %d\n", edges);
    // print_cuda_sum(weight_forward, edges, "After Matrix weight_forward");

    // cudaStreamSynchronize(stream);
    // print_cuda_sum(weight_forward, edges, "weight_forward");
    // print_cuda_sum(input_tran, column_num* feature_size, "input_tran");
    // print_cuda_sum(output_tran, batch_size * feature_size, "result");

//     print_cuda_sum(output, column_num * feature_size, "result");

//     cudaStreamSynchronize(stream);
#else
    printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Src\n");
       exit(0);
#endif

}
void Cuda_Stream::Gather_By_Dst_From_Src_with_cache(float* input,float* output,float* weight_forward,//data 
       VertexId_CUDA* cacheflag, VertexId_CUDA* destination,
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,//graph
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight,bool tensor_weight){
#if CUDA_ENABLE
        if(with_weight){
            if(tensor_weight){
//		aggregate_kernel_from_src_tensor_weight<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE,0,stream>>>(
//			row_indices, column_offset, input, output, weight_forward, 
//				src_start, dst_start, batch_size, feature_size);
                printf("aggregate_kernel_from_src_tensor_weight_optim_nts");
            }else{
                aggregate_kernel_from_src_with_weight_cache<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
                     cacheflag,destination,
			row_indices, column_offset, input, output, weight_forward, 
				src_start, dst_start, batch_size, feature_size);
            }
        }
        else{
                aggregate_kernel_from_src_without_weight<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
                        row_indices, column_offset, input, output, weight_forward, 
                                src_start, dst_start, batch_size, feature_size);
        }
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Src\n");
       exit(0);   
#endif  
        
}

void Cuda_Stream::Gather_By_Dst_From_Src_Optim(float* input,float* output,float* weight_forward,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight,bool tensor_weight){
#if CUDA_ENABLE
            if(with_weight){
            if(tensor_weight){
//		aggregate_kernel_from_src_tensor_weight_optim_nts<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
//			row_indices, column_offset, input, output, weight_forward, 
//				src_start,src_end, dst_start,dst_end,edges, batch_size, feature_size);
                printf("aggregate_kernel_from_src_tensor_weight_optim_nts is a legacy implementation\n");
                exit(0);  
            }else{
                aggregate_kernel_from_src_with_weight_optim_nts<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_indices, column_offset, input, output, weight_forward, 
				src_start,src_end, dst_start,dst_end,edges, batch_size, feature_size);
            }
        }
        else{
                aggregate_kernel_from_src_without_weight_optim_nts<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_indices, column_offset, input, output, weight_forward, 
				src_start,src_end, dst_start,dst_end,edges, batch_size, feature_size);
        }
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Src_Optim\n");
       exit(0);   
#endif      

    
}

void Cuda_Stream::Gather_By_Src_From_Dst_Optim(float* input,float* output,float* weight_forward,//data  
        VertexId_CUDA* row_offset,VertexId_CUDA *column_indices,
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight,bool tensor_weight){
#if CUDA_ENABLE
        if(with_weight){
            if(tensor_weight){
//		aggregate_kernel_from_dst_tensor_weight_optim_nts<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
//			row_offset, column_indices, input, output, weight_forward, 
//				src_start,src_end, dst_start,dst_end,edges, batch_size, feature_size);
                printf("aggregate_kernel_from_dst_tensor_weight_optim_nts is a legacy implementation\n");
                exit(0);  
            }else{
                aggregate_kernel_from_dst_with_weight_optim_nts<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_offset, column_indices, input, output, weight_forward, 
				src_start,src_end, dst_start,dst_end,edges, batch_size, feature_size);
            }
        }
        else{
                aggregate_kernel_from_dst_without_weight_optim_nts<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_offset, column_indices, input, output, weight_forward, 
				src_start,src_end, dst_start,dst_end,edges, batch_size, feature_size);
        }
#else
       printf("CUDA DISABLED Gather_By_Src_From_Dst_Optim\n");
       exit(0);   
#endif     
}


void Cuda_Stream::Gather_By_Src_From_Dst(float* input,float* output,float* weight_forward,//data 
        VertexId_CUDA* row_offset,VertexId_CUDA *column_indices,//graph
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	 VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight,bool tensor_weight){
#if CUDA_ENABLE
	//printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tTHREAD_SIZE:%d\n",BLOCK_SIZE,THREAD_SIZE); 
        if(with_weight){
            if(tensor_weight){
             printf("aggregate_kernel_from_dst_tensor_weight is a legacy implementation\n");
                exit(0);   
            }else{
                
		aggregate_kernel_from_dst_with_weight<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_offset, column_indices, input, output, weight_forward, 
				src_start, dst_start, batch_size, feature_size);   
            }
        }
        else{
                aggregate_kernel_from_dst_without_weight<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_offset, column_indices, input, output, weight_forward, 
				src_start, dst_start, batch_size, feature_size);
        }
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Src_From_Dst\n");
       exit(0);   
#endif 

}


void Cuda_Stream::Gather_By_Src_From_Dst_Spmm(float* input,float* output,float* weight_forward,//data 
        VertexId_CUDA* row_offset,VertexId_CUDA *column_indices, VertexId_CUDA column_num, //graph
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	 VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight,bool tensor_weight){
#if CUDA_ENABLE
       
       auto total_size = 0;
       if(!with_weight) {
              total_size += edges*sizeof(float);
       }
       if(cuda_buffer_size < total_size){
              auto new_size = total_size * 2;
              if(cuda_buffer_size != 0) {
                     cudaFree(cuda_buffer);
              }
              cudaMalloc((void**)&cuda_buffer, new_size);
              cuda_buffer_size = new_size;
       }

       // 为需要的空间分配buffer
       size_t buffer_cur = 0;
       if(!with_weight){
              weight_forward = (float*)cuda_buffer;
              buffer_cur += edges*sizeof(float);
       }
       // float* input_tran = (float*)(cuda_buffer + buffer_cur);
       // buffer_cur += feature_size * column_num * sizeof(float);
       // float* output_tran = (float*)(cuda_buffer + buffer_cur);
       // buffer_cur += batch_size*feature_size*sizeof(float);

       assert(sparse_handle != NULL);
       cusparseSpMatDescr_t matA;
       cusparseDnMatDescr_t matB, matC;
       void* dBuffer = NULL;
       size_t bufferSize = 0;
       float alpha = 1.0f;
       float beta  = 0.0f;
       // std::printf("edges: %d\n", edges);
       // std::printf("row: %d, column: %d\n", batch_size, column_num);
       if(!with_weight){
              // CHECK_CUDA_RESULT(cudaMallocAsync(&weight_forward, edges*sizeof(float), stream));
           // 0x0000803F 即1.0
           CHECK_CUDA_RESULT(cudaMemsetAsync(&weight_forward, 0x0000803F, sizeof(float) * edges, stream));
       }
//     cusparseHandle_t     sparse_handle = NULL;
//     CHECK_CUSPARSE( cusparseCreate(&sparse_handle) )
//     cusparseSetStream(sparse_handle, 0);

       // cudaStreamSynchronize(stream);
       // std::printf("column num: %d\n", column_num);
       // print_cuda_sum(row_indices, edges, "row_indices");
       // print_cuda_sum(column_offset, batch_size+1, "column offset");
       // print_cuda_sum(column_offset, batch_size+1, "column offset", 0);
       // print_cuda_sum(column_offset, batch_size+1, "column offset", batch_size);
       // print_cuda_sum(weight_forward, edges, "weight_forward");
       // print_cuda_sum(input, column_num*feature_size, "input");
       // print_cuda_sum(output, batch_size * feature_size, "output");
       // std::printf("CUDA v%d.%d\n", CUDART_VERSION/1000, CUDART_VERSION/10%100);
       CHECK_CUSPARSE(cusparseCreateCsr(&matA, batch_size, column_num, edges, row_offset, column_indices, weight_forward, 
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
       // float* input_tran;
       //  float *B = NULL;
       //  CHECK_CUDA_RESULT(cudaMalloc((void**)&input_tran, feature_size * column_num * sizeof(float)));
       // CHECK_CUBLAS(cublasSgeam( blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, column_num, feature_size, &alpha, input, feature_size, &beta, input, column_num, input_tran, column_num));
       // float* output_tran;
       // cudaMallocAsync((void**)&output_tran, batch_size*feature_size*sizeof(float), stream);
       // cudaMalloc((void**)&input_tran, column_num*feature_size*sizeof(float));
       // cudaMalloc((void**)&output_tran, batch_size*feature_size*sizeof(float));
       
       // print_cuda_sum(input_tran, column_num*feature_size, "input_tran");

       CHECK_CUSPARSE(cusparseCreateDnMat(&matB, column_num, feature_size, feature_size, input, CUDA_R_32F, CUSPARSE_ORDER_ROW));
       // CHECK_CUSPARSE(cusparseCreateDnMat(&matB, column_num, feature_size, feature_size, input, CUDA_R_32F, CUSPARSE_ORDER_ROW));
       CHECK_CUSPARSE(cusparseCreateDnMat(&matC, batch_size, feature_size, feature_size, output, CUDA_R_32F, CUSPARSE_ORDER_ROW));
       CHECK_CUSPARSE(cusparseSpMM_bufferSize(
              sparse_handle,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              &alpha, matA, matB, &beta, matC, CUDA_R_32F,
              CUSPARSE_SPMM_CSR_ALG2, &bufferSize
       ));
       total_size += bufferSize;
       if(buffer_cur + bufferSize < cuda_buffer_size) {
              dBuffer = cuda_buffer + buffer_cur;
              buffer_cur += bufferSize;
       } else {
              CHECK_CUDA_RESULT(cudaMallocAsync(&dBuffer, bufferSize, stream));
       }


       CHECK_CUSPARSE(cusparseSpMM(
              sparse_handle,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              &alpha, matA, matB, &beta, matC, CUDA_R_32F,
              CUSPARSE_SPMM_CSR_ALG2, dBuffer
       ));
       // 矩阵行列转换
       // CHECK_CUBLAS(cublasSgeam( blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, feature_size, batch_size, &alpha, output_tran, batch_size, &beta, output_tran, feature_size, output, feature_size));
       

       CHECK_CUSPARSE(cusparseDestroySpMat(matA));
       CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
       CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
       //     CHECK_CUSPARSE( cusparseDestroy(sparse_handle) )

       // cudaFreeAsync(dBuffer, stream);
       // cudaFreeAsync(input_tran, stream);
       // cudaFreeAsync(output_tran, stream);
       // if(!with_weight){
       //        cudaFreeAsync(weight_forward, stream);
       // }
       
       if(cuda_buffer_size < total_size){
              auto new_size = total_size * 2;
              cudaFreeAsync(dBuffer, stream);
              if(cuda_buffer_size != 0) {
                     cudaFree(cuda_buffer);
              }
              cudaMalloc((void**)&cuda_buffer, new_size);
              cuda_buffer_size = new_size;
       }

       // std::printf("edges: %d\n", edges);
       // print_cuda_sum(weight_forward, edges, "After Matrix weight_forward");

       // cudaStreamSynchronize(stream);
       // print_cuda_sum(weight_forward, edges, "weight_forward");
       // print_cuda_sum(input_tran, column_num* feature_size, "input_tran");
       // print_cuda_sum(output_tran, batch_size * feature_size, "result");
       
       // print_cuda_sum(output, batch_size * feature_size, "destroyed result");

       // cudaStreamSynchronize(stream);
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Src_From_Dst\n");
       exit(0);   
#endif 

}

void Cuda_Stream::Scatter_Grad_Back_To_Message(float* input,float* message_grad,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight){
#if CUDA_ENABLE
	//printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tTHREAD_SIZE:%d\n",BLOCK_SIZE,THREAD_SIZE); 
        if(with_weight){
            printf("tensor_weight Scatter_Grad_Back_To_Weight not implemented\n");
            exit(0);
        }else{
            scatter_grad_back_to_messaage<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_indices, column_offset, input, message_grad, 
				src_start, dst_start, batch_size, feature_size);
        }
#else
       printf("CUDA DISABLED Cuda_Stream::Scatter_Grad_Back_To_Message\n");
       exit(0);   
#endif 


}

void Cuda_Stream::Scatter_Src_Mirror_to_Msg(float* message,float* src_mirror_feature,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
        VertexId_CUDA* mirror_index, VertexId_CUDA batch_size,
        VertexId_CUDA feature_size){
#if CUDA_ENABLE
        scatter_src_mirror_to_msg<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            message, src_mirror_feature, row_indices, column_offset, mirror_index,
                batch_size, feature_size); 
#else
       printf("CUDA DISABLED Cuda_Stream::Scatter_Src_Mirror_to_Msg\n");
       exit(0);   
#endif
        
}
void Cuda_Stream::Scatter_Src_to_Msg(float* message,float* src_mirror_feature,//data
                                            VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
                                            VertexId_CUDA batch_size, VertexId_CUDA feature_size){
#if CUDA_ENABLE
    scatter_src_to_msg<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            message, src_mirror_feature, row_indices, column_offset,batch_size, feature_size);
#else
    printf("CUDA DISABLED Cuda_Stream::Scatter_Src_to_Msg\n");
       exit(0);
#endif

}

void Cuda_Stream::Scatter_Src_Dst_to_Msg(float* message,float* src_mirror_feature,//data
                                     VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
                                     VertexId_CUDA batch_size, VertexId_CUDA feature_size, VertexId_CUDA* dst_to_local){
#if CUDA_ENABLE
    scatter_src_dst_to_msg_map<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            message, src_mirror_feature, row_indices, column_offset,batch_size, feature_size, dst_to_local);
#else
    printf("CUDA DISABLED Cuda_Stream::Scatter_Src_to_Msg\n");
       exit(0);
#endif

}


void Cuda_Stream::Scatter_Src_to_Msg_Map(float* message,float* src_mirror_feature,//data
                                     VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
                                     VertexId_CUDA batch_size, VertexId_CUDA feature_size,
                                     VertexId_CUDA* src_to_local){
#if CUDA_ENABLE
    scatter_src_to_msg_map<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            message, src_mirror_feature, row_indices, column_offset,batch_size, feature_size,
            src_to_local);
#else
    printf("CUDA DISABLED Cuda_Stream::Scatter_Src_to_Msg_Map\n");
       exit(0);
#endif

}

void Cuda_Stream::Gather_Msg_To_Src_Mirror(float* src_mirror_feature,float* message,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
        VertexId_CUDA* mirror_index, VertexId_CUDA batch_size,
        VertexId_CUDA feature_size){
#if CUDA_ENABLE
	//printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tfeature_size:%d\n",BLOCK_SIZE,feature_size); 
        gather_msg_to_src_mirror<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            src_mirror_feature, message, row_indices, column_offset, mirror_index,
                batch_size, feature_size); 
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_Msg_To_Src_Mirror\n");
       exit(0);   
#endif
        
}


void Cuda_Stream::Gather_Msg_To_Src(float* src_mirror_feature,float* message,//data
                                           VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
                                           VertexId_CUDA batch_size, VertexId_CUDA feature_size){
#if CUDA_ENABLE
    //printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tfeature_size:%d\n",BLOCK_SIZE,feature_size);
    gather_msg_to_src<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            src_mirror_feature, message, row_indices, column_offset, batch_size, feature_size);
#else
    printf("CUDA DISABLED Cuda_Stream::Gather_Msg_To_Src_Mirror\n");
       exit(0);
#endif

}

void Cuda_Stream::Gather_Msg_To_Src_Dst(float* src_mirror_feature,float* message,//data
                                    VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
                                    VertexId_CUDA batch_size, VertexId_CUDA feature_size,
                                    VertexId_CUDA* dst_to_local){
#if CUDA_ENABLE
    //printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tfeature_size:%d\n",BLOCK_SIZE,feature_size);
    gather_msg_to_src_dst_map<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            src_mirror_feature, message, row_indices, column_offset, batch_size,
            feature_size, dst_to_local);
#else
    printf("CUDA DISABLED Cuda_Stream::Gather_Msg_To_Src_Mirror\n");
       exit(0);
#endif

}

void Cuda_Stream::Gather_Msg_To_Src_Map(float* src_mirror_feature,float* message,//data
                                    VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
                                    VertexId_CUDA batch_size, VertexId_CUDA feature_size,
                                    VertexId_CUDA* src_to_local){
#if CUDA_ENABLE
    //printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tfeature_size:%d\n",BLOCK_SIZE,feature_size);
    gather_msg_to_src_map<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            src_mirror_feature, message, row_indices, column_offset, batch_size, feature_size,
            src_to_local);
#else
    printf("CUDA DISABLED Cuda_Stream::Gather_Msg_To_Src_Map\n");
       exit(0);
#endif

}


void Cuda_Stream::Scatter_Dst_to_Msg(float* message,float* dst_feature,//data 
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size){
#if CUDA_ENABLE
	//printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tfeature_size:%d\n",BLOCK_SIZE,feature_size); 
        scatter_dst_to_msg<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            message, dst_feature, row_indices, column_offset,
            batch_size, feature_size); 
#else
       printf("CUDA DISABLED Cuda_Stream::Scatter_Dst_to_Msg\n");
       exit(0);   
#endif      
}

void Cuda_Stream::Scatter_Dst_to_Msg_Map(float* message,float* dst_feature,//data
                                     VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
                                     VertexId_CUDA batch_size, VertexId_CUDA feature_size,
                                     VertexId_CUDA* dst_to_local){
#if CUDA_ENABLE
    //printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tfeature_size:%d\n",BLOCK_SIZE,feature_size);
    scatter_dst_to_msg_map<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            message, dst_feature, row_indices, column_offset,
            batch_size, feature_size, dst_to_local);
#else
    printf("CUDA DISABLED Cuda_Stream::Scatter_Dst_to_Msg\n");
       exit(0);
#endif
}

void Cuda_Stream::Gather_Msg_to_Dst(float* dst_feature,float* message,//data 
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size){
#if CUDA_ENABLE
	//printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tfeature_size:%d\n",BLOCK_SIZE,feature_size); 
        gather_msg_to_dst<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            dst_feature, message, row_indices, column_offset,
            batch_size, feature_size); 
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_Msg_to_Dst\n");
       exit(0);   
#endif      
}

void Cuda_Stream::Gather_Msg_to_Dst_Map(float* dst_feature,float* message,//data
                                    VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
                                    VertexId_CUDA batch_size, VertexId_CUDA feature_size,
                                    VertexId_CUDA* dst_to_local){
#if CUDA_ENABLE
    //printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tfeature_size:%d\n",BLOCK_SIZE,feature_size);
    gather_msg_to_dst_map<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            dst_feature, message, row_indices, column_offset,
            batch_size, feature_size, dst_to_local);
#else
    printf("CUDA DISABLED Cuda_Stream::Gather_Msg_to_Dst\n");
       exit(0);
#endif
}

void Cuda_Stream::sample_processing_get_co_gpu(VertexId_CUDA *dst, 
                                   VertexId_CUDA *local_column_offset,
                                   VertexId_CUDA *global_column_offset,
                                   VertexId_CUDA dst_size,
                                   VertexId_CUDA* tmp_data_buffer,
                                   VertexId_CUDA src_index_size,
					VertexId_CUDA* src_count,
					VertexId_CUDA* src_index,
                                   VertexId_CUDA fanout,
                                   VertexId_CUDA & edge_size){
#if CUDA_ENABLE
       // 确定每个节点需要采的数量，方便为数组分配空间
    sample_processing_get_co_gpu_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                            (dst,tmp_data_buffer,global_column_offset,dst_size,
                            src_index_size,src_count,src_index,fanout);

//    this->CUDA_DEVICE_SYNCHRONIZE();
//    inclusiveTime -= get_time();
//    int num_items = dst_size + 1;
//    void *d_temp_storage = NULL;
//    size_t temp_storage_bytes = 0;
//    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, tmp_data_buffer, local_column_offset, num_items,stream);
//    if(temp_storage_bytes < cuda_buffer_size) {
//       d_temp_storage = cuda_buffer;
//    } else {
//       // std::printf("在这里进行了重分配\n\n\n");
//       if(cuda_buffer_size != 0) {
//              cudaFreeAsync(cuda_buffer, stream);
//       }
//       cuda_buffer_size = temp_storage_bytes * 2;
//       cudaMallocAsync((void**)&cuda_buffer, cuda_buffer_size, stream);
//       d_temp_storage = cuda_buffer;
//
//    }
////     cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
//    //printf("temp_storage_bytes:%d num_items:%d\n",temp_storage_bytes,num_items);
//    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, tmp_data_buffer, local_column_offset, num_items,stream);
//    this->CUDA_DEVICE_SYNCHRONIZE();
////     cudaFreeAsync(d_temp_storage, stream);
//    inclusiveTime += get_time();

//    inclusiveTime -= get_time();
//    thrust::device_ptr<VertexId_CUDA> g_arr = thrust::device_pointer_cast(tmp_data_buffer);
//    thrust::inclusive_scan(g_arr, g_arr + dst_size + 1, local_column_offset);
//    inclusiveTime += get_time();

    cpu_inclusiveTime -= get_time();
    size_t arr_size = (dst_size + 1) * sizeof(VertexId_CUDA);
    if(arr_size*2 > cpu_buffer_size) {
        if(cpu_buffer_size != 0) {
            delete []cpu_buffer;
        }
        cpu_buffer = new unsigned char[arr_size * 4];
        cpu_buffer_size = arr_size * 4;
    }
    VertexId_CUDA* cpu_data_buffer = (VertexId_CUDA*)(cpu_buffer);
    VertexId_CUDA* cpu_column_offset = (VertexId_CUDA*)(cpu_buffer +arr_size);
    cudaMemcpyAsync(cpu_data_buffer, tmp_data_buffer, arr_size, cudaMemcpyDeviceToHost, stream);
    cpu_column_offset[0] = cpu_data_buffer[0];
    for(int i = 1; i < dst_size + 1; i++) {
        cpu_column_offset[i] = cpu_data_buffer[i] + cpu_column_offset[i-1];
    }
    edge_size = cpu_column_offset[dst_size];
    cudaMemcpyAsync(local_column_offset, cpu_column_offset, arr_size, cudaMemcpyHostToDevice, stream);
//    delete []cpu_data_buffer;
//    delete []cpu_column_offset;
    cpu_inclusiveTime += get_time();

//    VertexId_CUDA* gpu_column_offset = cpu_data_buffer;
//    cudaMemcpyAsync(gpu_column_offset, local_column_offset, sizeof(VertexId_CUDA)*(dst_size + 1), cudaMemcpyDeviceToHost, stream);
//    bool correct = true;
//    for(int i = 0; i < dst_size + 1; i++) {
//        if(cpu_column_offset[i] != gpu_column_offset[i]) {
//            correct = false;
//            break;
//        }
//    }
//    if(correct) {
//        std::printf("cpu结果和GPU结果一致\n");
//    } else {
//        std::printf("cpu结果和GPU结果不一致\n");
//        exit(1);
//    }


#else
       printf("CUDA DISABLED Cuda_Stream::sample_processing_get_co_gpu\n");
       exit(0);   
#endif     
}

void Cuda_Stream::sample_processing_get_co_gpu_omit(
                                   VertexId_CUDA *CacheFlag,
                                   VertexId_CUDA *dst, 
                                   VertexId_CUDA *local_column_offset,
                                   VertexId_CUDA *global_column_offset,
                                   VertexId_CUDA dst_size,
                                   VertexId_CUDA* tmp_data_buffer,
                                   VertexId_CUDA src_index_size,
					VertexId_CUDA* src_count,
					VertexId_CUDA* src_index,
                                   VertexId_CUDA fanout,
                                   VertexId_CUDA & edge_size){
#if CUDA_ENABLE
       // 确定每个节点需要采的数量，方便为数组分配空间
//    print_cache_num<<<1,1>>>(dst, CacheFlag, dst_size);
     sample_processing_get_co_gpu_kernel_omit<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                             (CacheFlag,dst,tmp_data_buffer,global_column_offset,dst_size,
                             src_index_size,src_count,src_index,fanout);

    // NOTE: Toao用于检测cache点数量
    // TODO: Toao用于检测cache点的数量
//    VertexId_CUDA* cache_count;
//    cudaMallocAsync(&cache_count, sizeof(VertexId_CUDA), stream);
//    cudaMemsetAsync(cache_count, 0, sizeof(VertexId_CUDA), stream);
//    sample_processing_get_co_gpu_kernel_omit_lab<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
//                            (CacheFlag,dst,tmp_data_buffer,global_column_offset,dst_size,
//                            src_index_size,src_count,src_index,fanout, cache_count);
//    VertexId_CUDA* cache_count_cpu = new VertexId_CUDA[1]();
//    cudaMemcpyAsync(cache_count_cpu, cache_count, sizeof(VertexId_CUDA), cudaMemcpyDeviceToHost, stream);
////    std::printf("采样总结点数: %u, cache点数量: %u\n", dst_size, cache_count_cpu[0]);
//    total_sample_num += dst_size;
//    total_cache_hit += cache_count_cpu[0];
//    delete []cache_count_cpu;
//    cudaFreeAsync(cache_count, stream);

//    this->CUDA_DEVICE_SYNCHRONIZE();
//    inclusiveTime -= get_time();
//    int num_items = dst_size + 1;
//    void *d_temp_storage = NULL;
//    size_t temp_storage_bytes = 0;
//    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, tmp_data_buffer, local_column_offset, num_items,stream);
//    if(temp_storage_bytes < cuda_buffer_size) {
//       d_temp_storage = cuda_buffer;
//    } else {
//       // std::printf("在这里进行了重分配\n\n\n");
//       if(cuda_buffer_size != 0) {
//              cudaFreeAsync(cuda_buffer, stream);
//       }
//       cuda_buffer_size = temp_storage_bytes * 2;
//       cudaMallocAsync((void**)&cuda_buffer, cuda_buffer_size, stream);
//       d_temp_storage = cuda_buffer;
//
//    }
////     cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
//    //printf("temp_storage_bytes:%d num_items:%d\n",temp_storage_bytes,num_items);
//    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, tmp_data_buffer, local_column_offset, num_items,stream);
//    this->CUDA_DEVICE_SYNCHRONIZE();
////     cudaFreeAsync(d_temp_storage, stream);
//    inclusiveTime += get_time();

//    inclusiveTime -= get_time();
//    thrust::device_ptr<VertexId_CUDA> g_arr = thrust::device_pointer_cast(tmp_data_buffer);
//    thrust::inclusive_scan(g_arr, g_arr + dst_size + 1, local_column_offset);
//    inclusiveTime += get_time();

    cpu_inclusiveTime -= get_time();
    size_t arr_size = (dst_size + 1) * sizeof(VertexId_CUDA);
    if(arr_size*2 > cpu_buffer_size) {
        if(cpu_buffer_size != 0) {
            delete []cpu_buffer;
        }
        cpu_buffer = new unsigned char[arr_size * 4];
        cpu_buffer_size = arr_size * 4;
    }
    VertexId_CUDA* cpu_data_buffer = (VertexId_CUDA*)(cpu_buffer);
    VertexId_CUDA* cpu_column_offset = (VertexId_CUDA*)(cpu_buffer +arr_size);
    cudaMemcpyAsync(cpu_data_buffer, tmp_data_buffer, arr_size, cudaMemcpyDeviceToHost, stream);
    cpu_column_offset[0] = cpu_data_buffer[0];
    for(int i = 1; i < dst_size + 1; i++) {
        cpu_column_offset[i] = cpu_data_buffer[i] + cpu_column_offset[i-1];
    }
    edge_size = cpu_column_offset[dst_size];
    cudaMemcpyAsync(local_column_offset, cpu_column_offset, arr_size, cudaMemcpyHostToDevice, stream);

//    delete []cpu_data_buffer;
//    delete []cpu_column_offset;
    cpu_inclusiveTime += get_time();

//    VertexId_CUDA* gpu_column_offset = cpu_data_buffer;
//    cudaMemcpyAsync(gpu_column_offset, local_column_offset, sizeof(VertexId_CUDA)*(dst_size + 1), cudaMemcpyDeviceToHost, stream);
//    bool correct = true;
//    for(int i = 0; i < dst_size + 1; i++) {
//        if(cpu_column_offset[i] != gpu_column_offset[i]) {
//            correct = false;
//            break;
//        }
//    }
//    if(correct) {
//        std::printf("cpu结果和GPU结果一致\n");
//    } else {
//        std::printf("cpu结果和GPU结果不一致\n");
//        exit(1);
//    }


#else
       printf("CUDA DISABLED Cuda_Stream::sample_processing_get_co_gpu\n");
       exit(0);   
#endif     
}


void Cuda_Stream::sample_processing_get_co_gpu_omit(
        VertexId_CUDA *CacheFlag,
        VertexId_CUDA *dst,
        VertexId_CUDA *local_column_offset,
        VertexId_CUDA *global_column_offset,
        VertexId_CUDA dst_size,
        VertexId_CUDA* tmp_data_buffer,
        VertexId_CUDA src_index_size,
        VertexId_CUDA* src_count,
        VertexId_CUDA* src_index,
        VertexId_CUDA fanout,
        VertexId_CUDA & edge_size,
        VertexId_CUDA super_batch_id){
#if CUDA_ENABLE
    // 确定每个节点需要采的数量，方便为数组分配空间
//    print_cache_num<<<1,1>>>(dst, CacheFlag, dst_size);
    sample_processing_get_co_gpu_kernel_omit<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
            (CacheFlag,dst,tmp_data_buffer,global_column_offset,dst_size,
             src_index_size,src_count,src_index,fanout, super_batch_id);

    // NOTE: Toao用于检测cache点数量
    // TODO: Toao用于检测cache点的数量
   VertexId_CUDA* cache_count;
   cudaMallocAsync(&cache_count, sizeof(VertexId_CUDA), stream);
   cudaMemsetAsync(cache_count, 0, sizeof(VertexId_CUDA), stream);
   sample_processing_get_co_gpu_kernel_omit_lab<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                           (CacheFlag,dst,tmp_data_buffer,global_column_offset,dst_size,
                           src_index_size,src_count,src_index,fanout, cache_count, super_batch_id);
   VertexId_CUDA* cache_count_cpu = new VertexId_CUDA[1]();
   cudaMemcpyAsync(cache_count_cpu, cache_count, sizeof(VertexId_CUDA), cudaMemcpyDeviceToHost, stream);
//    std::printf("采样总结点数: %u, cache点数量: %u\n", dst_size, cache_count_cpu[0]);
   total_sample_num += dst_size;
   total_cache_hit += cache_count_cpu[0];
   delete []cache_count_cpu;
   cudaFreeAsync(cache_count, stream);

//    this->CUDA_DEVICE_SYNCHRONIZE();
//    inclusiveTime -= get_time();
//    int num_items = dst_size + 1;
//    void *d_temp_storage = NULL;
//    size_t temp_storage_bytes = 0;
//    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, tmp_data_buffer, local_column_offset, num_items,stream);
//    if(temp_storage_bytes < cuda_buffer_size) {
//       d_temp_storage = cuda_buffer;
//    } else {
//       // std::printf("在这里进行了重分配\n\n\n");
//       if(cuda_buffer_size != 0) {
//              cudaFreeAsync(cuda_buffer, stream);
//       }
//       cuda_buffer_size = temp_storage_bytes * 2;
//       cudaMallocAsync((void**)&cuda_buffer, cuda_buffer_size, stream);
//       d_temp_storage = cuda_buffer;
//
//    }
////     cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
//    //printf("temp_storage_bytes:%d num_items:%d\n",temp_storage_bytes,num_items);
//    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, tmp_data_buffer, local_column_offset, num_items,stream);
//    this->CUDA_DEVICE_SYNCHRONIZE();
////     cudaFreeAsync(d_temp_storage, stream);
//    inclusiveTime += get_time();

//    inclusiveTime -= get_time();
//    thrust::device_ptr<VertexId_CUDA> g_arr = thrust::device_pointer_cast(tmp_data_buffer);
//    thrust::inclusive_scan(g_arr, g_arr + dst_size + 1, local_column_offset);
//    inclusiveTime += get_time();

    cpu_inclusiveTime -= get_time();
    size_t arr_size = (dst_size + 1) * sizeof(VertexId_CUDA);
    if(arr_size*2 > cpu_buffer_size) {
        if(cpu_buffer_size != 0) {
            delete []cpu_buffer;
        }
        cpu_buffer = new unsigned char[arr_size * 4];
        cpu_buffer_size = arr_size * 4;
    }
    VertexId_CUDA* cpu_data_buffer = (VertexId_CUDA*)(cpu_buffer);
    VertexId_CUDA* cpu_column_offset = (VertexId_CUDA*)(cpu_buffer +arr_size);
    cudaMemcpyAsync(cpu_data_buffer, tmp_data_buffer, arr_size, cudaMemcpyDeviceToHost, stream);
    cpu_column_offset[0] = cpu_data_buffer[0];
    for(int i = 1; i < dst_size + 1; i++) {
        cpu_column_offset[i] = cpu_data_buffer[i] + cpu_column_offset[i-1];
    }
    edge_size = cpu_column_offset[dst_size];
    cudaMemcpyAsync(local_column_offset, cpu_column_offset, arr_size, cudaMemcpyHostToDevice, stream);

//    delete []cpu_data_buffer;
//    delete []cpu_column_offset;
    cpu_inclusiveTime += get_time();

//    VertexId_CUDA* gpu_column_offset = cpu_data_buffer;
//    cudaMemcpyAsync(gpu_column_offset, local_column_offset, sizeof(VertexId_CUDA)*(dst_size + 1), cudaMemcpyDeviceToHost, stream);
//    bool correct = true;
//    for(int i = 0; i < dst_size + 1; i++) {
//        if(cpu_column_offset[i] != gpu_column_offset[i]) {
//            correct = false;
//            break;
//        }
//    }
//    if(correct) {
//        std::printf("cpu结果和GPU结果一致\n");
//    } else {
//        std::printf("cpu结果和GPU结果不一致\n");
//        exit(1);
//    }


#else
    printf("CUDA DISABLED Cuda_Stream::sample_processing_get_co_gpu\n");
       exit(0);
#endif
}

__global__ void print_cuda_array(VertexId_CUDA* array, VertexId_CUDA len) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid == 0) {
        for(int i = 0; i < len; i++) {
            std::printf("%u ", array[i]);
        }
        std::printf("\n");
    }
}

void print_cuda_array(char* message, VertexId_CUDA* gpu_arr, VertexId_CUDA len) {
    VertexId_CUDA *cpu_arr = new VertexId_CUDA [len];
    cudaMemcpy(cpu_arr, gpu_arr, sizeof(int)*len, cudaMemcpyDeviceToHost);
    std::printf("%s: ", message);
    for(int i = 0; i < len; i++) {
        std::printf("%u ", cpu_arr[i]);
    }
    std::printf("\n");
    delete []cpu_arr;
}



void Cuda_Stream::sample_processing_traverse_gpu(VertexId_CUDA *destination,
                                                 VertexId_CUDA *c_o,
							VertexId_CUDA *r_i,
							VertexId_CUDA *global_c_o,
							VertexId_CUDA *global_r_i,
							VertexId_CUDA *src_index,
                                   	       VertexId_CUDA vtx_size,
							VertexId_CUDA edge_size,
							VertexId_CUDA src_index_size,
							VertexId_CUDA* src,
						       VertexId_CUDA* src_count,
                             VertexId_CUDA layer,
                             VertexId_CUDA max_sample_count,
                             bool add_dst_to_src ){
#if CUDA_ENABLE
//    std::printf("vtx num: %u\n", vtx_size);
//    if(layer == 0) {
//        std::printf("实际第%d层采样节点\n", layer);
//        print_cuda_array<<<1, 1>>>(destination, vtx_size);
//        cudaDeviceSynchronize();
//        std::printf("\n");
//    }

    std::random_device rd;
    auto seed = rd();
    if(max_sample_count < 32) {
        sample_processing_traverse_gpu_kernel_stage2<32, 1><<<vtx_size,32,0,stream>>>(
                r_i,c_o,destination, vtx_size,global_c_o,global_r_i,src_index,
                max_sample_count,layer, seed);
    } else if(max_sample_count < 128) {
        sample_processing_traverse_gpu_kernel_stage2<64, 2><<<vtx_size,64,0,stream>>>(
                r_i,c_o,destination, vtx_size,global_c_o,global_r_i,src_index,
                max_sample_count,layer, seed);
    } else if(max_sample_count < 1024){
        sample_processing_traverse_gpu_kernel_stage2<CUDA_NUM_BLOCKS, 4><<<vtx_size,CUDA_NUM_BLOCKS,0,stream>>>(
                r_i,c_o,destination, vtx_size,global_c_o,global_r_i,src_index,
                max_sample_count,layer, seed);

    } else {
        sample_processing_traverse_gpu_kernel_stage2<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                (destination, c_o,r_i,global_c_o,global_r_i,src_index,vtx_size,layer);
    }

    // toao 注释掉同步
//    std::printf("after sample_processing_traverse_gpu_kernel_stage2\n");
//    this->CUDA_DEVICE_SYNCHRONIZE();
//    std::printf("before sample_processing_traverse_gpu_kernel_stage2\n");

//    sample_processing_traverse_gpu_kernel_stage2<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
//            (destination, c_o,r_i,global_c_o,global_r_i,src_index,vtx_size,layer);
//    VertexId_CUDA count = 0;
//    VertexId_CUDA* gpu_count;
//    cudaMallocAsync((void**)&gpu_count, sizeof(VertexId_CUDA), stream);
////    cudaMemcpyAsync(gpu_count, &count, sizeof(VertexId_CUDA), cudaMemcpyHostToDevice, stream);
//    check_sample<<<vtx_size, 128, 0, stream>>>(c_o, r_i, global_c_o, global_r_i, vtx_size, destination, gpu_count);
//    cudaMemcpyAsync(&count, gpu_count, sizeof(VertexId_CUDA), cudaMemcpyDeviceToHost, stream);
//    cudaDeviceSynchronize();
//    std::printf("不存在的邻居的个数: %u\n", count);
    if(add_dst_to_src) {
        sample_add_dst_to_src<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS, 0, stream>>>(src_index, destination,
                                                                                vtx_size, layer);
    }

    int block_num = (src_index_size / CUDA_NUM_THREADS) + 1;
    block_num = std::max(block_num, CUDA_NUM_BLOCKS);
    sample_processing_traverse_gpu_kernel_stage3<<<block_num,CUDA_NUM_THREADS,0,stream>>>
                            (src_index,src_index_size,src,src_count,layer);
//    this->CUDA_DEVICE_SYNCHRONIZE();
//    sample_indices << "下一层采样顶点数量: " << get_cuda_array_num(src_count, 0) << "\n";
//     this->CUDA_DEVICE_SYNCHRONIZE();

#else
       printf("CUDA DISABLED Cuda_Stream::sample_processing_traverse_gpu\n");
       exit(0);   
#endif   
}

void Cuda_Stream::sample_processing_update_ri_gpu(VertexId_CUDA *r_i,
						VertexId_CUDA *src_index,
                                   	VertexId_CUDA edge_size,
                                          VertexId_CUDA src_index_size){
#if CUDA_ENABLE
    sample_processing_update_ri_gpu_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                            (r_i,src_index,edge_size,src_index_size);
//    this->CUDA_DEVICE_SYNCHRONIZE();
//    cudaStreamSynchronize()
#else
    printf("CUDA DISABLED Cuda_Stream::sample_processing_update_ri_gpu\n"); 
    exit(0);   
#endif   
}

void Cuda_Stream::set_total_local_index(VertexId_CUDA* vtx_index, size_t vtx_size, VertexId_CUDA* vtx_count,
                                        VertexId_CUDA* dev_source, size_t source_size,
                                        VertexId_CUDA* dev_destination, size_t destination_size,
                                        VertexId_CUDA* dev_local_to_global, VertexId_CUDA* dev_src_to_local,
                                        VertexId_CUDA* dev_dst_to_local) {
    // 1. 先初始化src_index为-1，整型最大值
    cudaSetMemAsync(vtx_index, -1, sizeof(VertexId_CUDA) * vtx_size, stream);
    cudaSetMemAsync(vtx_count, 0, sizeof(VertexId_CUDA), stream);
    // 2. 标记src_index看dst和src中哪些顶点出现了，标记为1，利用source和destination数组
    sample_mark_src_dst<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS, 0, stream>>>(vtx_index, dev_source, source_size,
                                                               dev_destination, destination_size);
    // 3. 利用上面的标记并结合atomicAdd得出local_to_global的数组，同时src_index的标记也改为其在数组中的位置
    sample_set_local_to_global<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS, 0, stream>>>(vtx_index, vtx_size, vtx_count,
                                                                      dev_local_to_global);
    // 4. 利用src_index标记遍历source和destination建立dst_to_local和src_to_local
    sample_set_src_dst_local<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS, 0, stream>>>(vtx_index, dev_source, source_size, dev_destination,
                                                                    destination_size, dev_src_to_local, dev_dst_to_local);

}

void Cuda_Stream::set_dst_local_index(VertexId_CUDA* vtx_index, VertexId_CUDA* dev_destination,
                                      size_t destination_size, VertexId_CUDA* dev_dst_to_local) {
    // 结合destination遍历src_index得出dst在src中的标记位置
    sample_set_dst_local<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS, 0, stream>>>(vtx_index, dev_destination,
                                                                           destination_size, dev_dst_to_local);

}


void Cuda_Stream::check_dst_local_index(VertexId_CUDA* dev_dst_to_local,size_t destination_size,
                                        VertexId_CUDA src_size) {
    sample_check_dst_local<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS, 0, stream>>>(dev_dst_to_local, destination_size, src_size);

}

void Cuda_Stream::zero_copy_feature_move_gpu(float *dev_feature,
						float *pinned_host_feature,
						VertexId_CUDA *src_vertex,
                                   	VertexId_CUDA feature_size,
						VertexId_CUDA vertex_size){
#if CUDA_ENABLE
    // std::printf("传输feature大小数量: %u\n", vertex_size);
    total_transfer_node += vertex_size;
    // std::printf("total_transfer_node传输feature大小数量: %u\n", total_transfer_node);
    int block_num = ((vertex_size * WARP_SIZE)/CUDA_NUM_THREADS) + 1;
    block_num = std::min(CUDA_NUM_THREADS, block_num);
    zero_copy_feature_move_gpu_kernel<<<block_num/2,CUDA_NUM_THREADS,0,stream>>>
                            (dev_feature,pinned_host_feature,src_vertex,feature_size,vertex_size);
//     this->CUDA_DEVICE_SYNCHRONIZE();
#else
       printf("CUDA DISABLED Cuda_Stream::zero_copy_feature_move_gpu\n");
       exit(0);   
#endif   
}

// void Cuda_Stream::init_cache_map(VertexId_CUDA *src_vertex, VertexId_CUDA *cache_node_hashmap, 
//                                  VertexId_CUDA* vertex_size, VertexId_CUDA *local_idx, 
//                                  VertexId_CUDA * local_idx_cache) {
// #if CUDA_ENABLE
//   init_cache_map_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS, 0, stream>>>(
//       src_vertex, cache_node_hashmap, vertex_size, local_idx, local_idx_cache);
//   this->CUDA_DEVICE_SYNCHRONIZE();
// #else
//   printf("CUDA DISABLED Cuda_Stream::init_cache_map\n");
//   exit(0);
// #endif
// }

void Cuda_Stream::zero_copy_feature_move_gpu_cache(float* dev_feature, float* host_pinned_feature,
                                                   VertexId_CUDA* src_vertex, VertexId_CUDA feature_size,
                                                   VertexId_CUDA vertex_size, VertexId_CUDA* local_idx) {
#if CUDA_ENABLE
    total_transfer_node += vertex_size;
  zero_copy_feature_move_gpu_cache_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS, 0, stream>>>(
      dev_feature, host_pinned_feature, src_vertex, feature_size, vertex_size, local_idx);
  this->CUDA_DEVICE_SYNCHRONIZE();
#else
  printf("CUDA DISABLED Cuda_Stream::zero_copy_feature_move_gpu\n");
  exit(0);
#endif
}

void Cuda_Stream::gather_feature_from_gpu_cache(float* dev_feature, float* dev_cache_feature, VertexId_CUDA* src_vertex,
                                                VertexId_CUDA feature_size, VertexId_CUDA vertex_size,
                                                VertexId_CUDA* local_idx, VertexId_CUDA* cache_node_hashmap) {
  //  std::vector<int>& local_idx, std::vector<int>& cache_node_hashmap) {
#if CUDA_ENABLE
  gather_feature_from_gpu_cache_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS, 0, stream>>>(
      dev_feature, dev_cache_feature, src_vertex, feature_size, vertex_size, local_idx, cache_node_hashmap);
  this->CUDA_DEVICE_SYNCHRONIZE();
#else
  printf("CUDA DISABLED Cuda_Stream::zero_copy_feature_move_gpu\n");
  exit(0);
#endif
}




void Cuda_Stream::zero_copy_embedding_move_gpu(float *dev_feature,
						float *pinned_host_feature,
                                   	VertexId_CUDA feature_size,
						VertexId_CUDA vertex_size){
#if CUDA_ENABLE
    zero_copy_embedding_move_gpu_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                            (dev_feature,pinned_host_feature,feature_size,vertex_size);
    this->CUDA_DEVICE_SYNCHRONIZE();
#else
       printf("CUDA DISABLED Cuda_Stream::zero_copy_feature_move_gpu\n");
       exit(0);   
#endif   
}

void Cuda_Stream::global_copy_label_move_gpu(long *dev_label,
				long *global_dev_label,
				VertexId_CUDA *dst_vertex,
				VertexId_CUDA vertex_size){
#if CUDA_ENABLE
    int block_num = (vertex_size / CUDA_NUM_THREADS) + 1;
    block_num = std::min(block_num, CUDA_NUM_BLOCKS);
    global_copy_label_move_gpu_kernel<<<block_num,CUDA_NUM_THREADS,0,stream>>>
                            (dev_label,global_dev_label,dst_vertex,vertex_size);
//     this->CUDA_DEVICE_SYNCHRONIZE();
#else
       printf("CUDA DISABLED Cuda_Stream::copy_label_from_global\n");
       exit(0);   
#endif   
}

void Cuda_Stream::dev_load_share_embedding(float *dev_embedding,
				float *share_embedding,
				VertexId_CUDA *dev_cacheflag,
                            VertexId_CUDA *dev_cachemap,
                            VertexId_CUDA feature_size,
                            VertexId_CUDA *destination_vertex,
				VertexId_CUDA vertex_size){
#if CUDA_ENABLE
    dev_load_share_embedding_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                            (dev_embedding,share_embedding,dev_cacheflag,dev_cachemap,feature_size,destination_vertex,vertex_size);
    this->CUDA_DEVICE_SYNCHRONIZE();
#else
       printf("CUDA DISABLED Cuda_Stream::copy_label_from_global\n");
       exit(0);   
#endif   
}


void Cuda_Stream::dev_load_share_embedding_and_feature(float* dev_feature, float *dev_embedding,
                                           float* share_feature, float *share_embedding,
                                           VertexId_CUDA *dev_cacheflag,
                                           VertexId_CUDA *dev_cachemap,
                                           VertexId_CUDA feature_size, VertexId_CUDA embedding_size,
                                           VertexId_CUDA *destination_vertex,
                                           VertexId_CUDA vertex_size){
#if CUDA_ENABLE
//    print_cache_num<<<1, 1>>>(destination_vertex, dev_cachemap, vertex_size);
//    this->CUDA_DEVICE_SYNCHRONIZE();

    dev_load_share_embedding_and_feature_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
            (dev_feature, dev_embedding, share_feature, share_embedding,dev_cacheflag,dev_cachemap,
             feature_size,embedding_size, destination_vertex,vertex_size);
    this->CUDA_DEVICE_SYNCHRONIZE();
//    cudaDeviceSynchronize();
#else
    printf("CUDA DISABLED Cuda_Stream::copy_label_from_global\n");
       exit(0);
#endif
}


void Cuda_Stream::dev_load_share_embedding_and_feature(float* dev_feature, float *dev_embedding,
                                                       float* share_feature, float *share_embedding,
                                                       VertexId_CUDA *dev_cacheflag,
                                                       VertexId_CUDA *dev_cachelocation,
                                                       VertexId_CUDA feature_size, VertexId_CUDA embedding_size,
                                                       VertexId_CUDA *destination_vertex,
                                                       VertexId_CUDA vertex_size, VertexId_CUDA super_batch_id){
#if CUDA_ENABLE
//    print_cache_num<<<1, 1>>>(destination_vertex, dev_cachemap, vertex_size);
//    this->CUDA_DEVICE_SYNCHRONIZE();

    dev_load_share_embedding_and_feature_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
            (dev_feature, dev_embedding, share_feature, share_embedding,dev_cacheflag,dev_cachelocation,
             feature_size,embedding_size, destination_vertex,vertex_size, super_batch_id);
    this->CUDA_DEVICE_SYNCHRONIZE();
//    cudaDeviceSynchronize();
#else
    printf("CUDA DISABLED Cuda_Stream::copy_label_from_global\n");
       exit(0);
#endif
}


void Cuda_Stream::dev_load_share_embedding(float *dev_embedding,
                                                       float *share_embedding,
                                                       VertexId_CUDA *dev_cacheflag,
                                                       VertexId_CUDA *dev_cachelocation,
                                                       VertexId_CUDA embedding_size,
                                                       VertexId_CUDA *destination_vertex,
                                                       VertexId_CUDA vertex_size, VertexId_CUDA super_batch_id){
#if CUDA_ENABLE
//    print_cache_num<<<1, 1>>>(destination_vertex, dev_cachemap, vertex_size);
//    this->CUDA_DEVICE_SYNCHRONIZE();

    dev_load_share_embedding_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
            (dev_embedding, share_embedding,dev_cacheflag,dev_cachelocation,
             embedding_size, destination_vertex,vertex_size, super_batch_id);
    this->CUDA_DEVICE_SYNCHRONIZE();
//    cudaDeviceSynchronize();
#else
    printf("CUDA DISABLED Cuda_Stream::copy_label_from_global\n");
       exit(0);
#endif
}

void Cuda_Stream::dev_load_share_aggregate(float* dev_feature,
                                                       float* share_feature,
                                                       VertexId_CUDA *dev_cacheflag,
                                                       VertexId_CUDA *dev_cachemap,
                                                       VertexId_CUDA feature_size,
                                                       VertexId_CUDA *destination_vertex,
                                                       VertexId_CUDA vertex_size){
#if CUDA_ENABLE
//    print_cache_num<<<1, 1>>>(destination_vertex, dev_cachemap, vertex_size);
//    this->CUDA_DEVICE_SYNCHRONIZE();

    dev_load_share_aggregate_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
            (dev_feature, share_feature,dev_cacheflag,dev_cachemap,
             feature_size,destination_vertex,vertex_size);
    this->CUDA_DEVICE_SYNCHRONIZE();
//    cudaDeviceSynchronize();
#else
    printf("CUDA DISABLED Cuda_Stream::copy_label_from_global\n");
       exit(0);
#endif
}

void Cuda_Stream::dev_get_X_mask(uint8_t* dev_X_mask,
                                           VertexId_CUDA *destination,
                                           VertexId_CUDA *dev_cacheflag,
                                           VertexId_CUDA vertex_size){
#if CUDA_ENABLE
//    print_cache_num<<<1, 1>>>(destination_vertex, dev_cachemap, vertex_size);
//    this->CUDA_DEVICE_SYNCHRONIZE();
    dev_get_X_mask_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(dev_X_mask, destination, dev_cacheflag, vertex_size);
    this->CUDA_DEVICE_SYNCHRONIZE();
//    cudaDeviceSynchronize();
#else
    printf("CUDA DISABLED Cuda_Stream::copy_label_from_global\n");
       exit(0);
#endif
}

void Cuda_Stream::dev_print_avg_weight(VertexId_CUDA* column_offset, VertexId_CUDA *row_indices,float* weight, VertexId_CUDA *destination,
                                       VertexId_CUDA* dev_cacheflag,float* dev_sum, VertexId_CUDA* dev_cache_num, VertexId_CUDA vertex_size){
    dev_print_avg_weight_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS, 0, stream>>>
        (column_offset, row_indices,weight, destination, dev_cacheflag, dev_sum, dev_cache_num, vertex_size);
}


void Cuda_Stream::dev_load_share_embedding(float *dev_embedding,
                                           float *share_embedding,
                                           VertexId_CUDA *dev_cacheflag,
                                           VertexId_CUDA *dev_cachemap,
                                           VertexId_CUDA feature_size,
                                           VertexId_CUDA *destination_vertex,
                                           uint8_t *dev_x_mask,
                                           uint8_t *dev_cache_mask,
                                           VertexId_CUDA vertex_size){
#if CUDA_ENABLE
    dev_load_share_embedding_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
            (dev_embedding,share_embedding,dev_cacheflag,dev_cachemap,feature_size,destination_vertex,dev_x_mask, dev_cache_mask, vertex_size);
    this->CUDA_DEVICE_SYNCHRONIZE();
#else
    printf("CUDA DISABLED Cuda_Stream::copy_label_from_global\n");
       exit(0);
#endif
}


void Cuda_Stream::dev_Grad_accumulate(float *dev_grad_buffer,
				float *dev_share_grad,
				VertexId_CUDA *dev_cacheflag,
                            VertexId_CUDA *dev_cachemap,
                            VertexId_CUDA feature_size,
                            VertexId_CUDA *destination_vertex,
				VertexId_CUDA vertex_size){
#if CUDA_ENABLE
    dev_Grad_accumulate_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                            (dev_grad_buffer,dev_share_grad,dev_cacheflag,dev_cachemap,feature_size,destination_vertex,vertex_size);
    this->CUDA_DEVICE_SYNCHRONIZE();
#else
       printf("CUDA DISABLED Cuda_Stream::copy_label_from_global\n");
       exit(0);   
#endif   
}

void Cuda_Stream::dev_update_share_embedding(float *dev_embedding,
				float *share_embedding,
                            VertexId_CUDA *dev_cachemap,
				VertexId_CUDA *dev_cacheflag,
                            VertexId_CUDA feature_size,
                            VertexId_CUDA *destination_vertex,
				VertexId_CUDA vertex_size){
#if CUDA_ENABLE
    dev_update_share_embedding_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                            (dev_embedding,share_embedding,dev_cachemap,dev_cacheflag,feature_size,destination_vertex,vertex_size);
    this->CUDA_DEVICE_SYNCHRONIZE();
#else
       printf("CUDA DISABLED Cuda_Stream::dev_update_share_embedding\n");
       exit(0);   
#endif   
}

void Cuda_Stream::dev_update_share_embedding_and_feature(float *dev_feature,
                                                         float *dev_embedding,
                                                         float * share_aggregate,
                                             float *share_embedding,
                                             VertexId_CUDA *dev_cachemap,
                                             VertexId_CUDA *dev_cacheflag,
                                             VertexId_CUDA feature_size,
                                             VertexId_CUDA embedding_size,
                                             VertexId_CUDA *destination_vertex,
                                             VertexId_CUDA *dev_X_version,
                                             VertexId_CUDA *dev_Y_version,
                                             VertexId_CUDA vertex_size, VertexId_CUDA require_version){
#if CUDA_ENABLE
//    print_cache_num<<<1, 1>>>(destination_vertex, dev_cachemap, vertex_size);
//    this->CUDA_DEVICE_SYNCHRONIZE();
    dev_update_share_embedding_and_feature_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
            (dev_feature,dev_embedding,share_aggregate, share_embedding,dev_cachemap,dev_cacheflag,feature_size,
             embedding_size, destination_vertex, dev_X_version, dev_Y_version, vertex_size, require_version);
    this->CUDA_DEVICE_SYNCHRONIZE();
#else
    printf("CUDA DISABLED Cuda_Stream::dev_update_share_embedding\n");
       exit(0);
#endif
}

void Cuda_Stream::ReFreshDegree(VertexId_CUDA *out_degree,
				    VertexId_CUDA *in_degree,
				    VertexId_CUDA vertices){
#if CUDA_ENABLE
    re_fresh_degree<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                                                 (out_degree,in_degree,vertices);
    this->CUDA_DEVICE_SYNCHRONIZE();
#else
       printf("CUDA DISABLED Cuda_Stream::re_fresh_degree\n");
       exit(0);   
#endif   
}

void Cuda_Stream::UpdateDegree(VertexId_CUDA *out_degree,
				   VertexId_CUDA *in_degree,
				   VertexId_CUDA vertices,
                               VertexId_CUDA *destination,
                               VertexId_CUDA *source,
                               VertexId_CUDA *column_offset,
				   VertexId_CUDA *row_indices){
#if CUDA_ENABLE
    up_date_degree<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                                          (out_degree,in_degree,vertices,destination,source,column_offset,row_indices);
//    this->CUDA_DEVICE_SYNCHRONIZE();
#else
       printf("CUDA DISABLED Cuda_Stream::up_date_degree\n");
       exit(0);   
#endif   
}


void Cuda_Stream::UpdateDegreeCache(VertexId_CUDA *out_degree,
                               VertexId_CUDA *in_degree,
                               VertexId_CUDA vertices,
                               VertexId_CUDA *destination,
                               VertexId_CUDA *source,
                               VertexId_CUDA *column_offset,
                               VertexId_CUDA *row_indices,
                               int fanout){
#if CUDA_ENABLE
    update_cache_degree<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
            (out_degree,in_degree,vertices,destination,source,column_offset,row_indices, fanout);
//    this->CUDA_DEVICE_SYNCHRONIZE();
#else
    printf("CUDA DISABLED Cuda_Stream::up_date_degree\n");
       exit(0);
#endif
}

void Cuda_Stream::move_degree_to_gpu(VertexId_CUDA* cpu_in_degree, VertexId_CUDA* cpu_out_degree,
                                     VertexId_CUDA *gpu_in_degree, VertexId_CUDA *gpu_out_degree,VertexId_CUDA vertexs) {
#if CUDA_ENABLE
    cudaMemcpyAsync(gpu_in_degree, cpu_in_degree, sizeof(VertexId_CUDA) * vertexs, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu_out_degree, cpu_in_degree, sizeof(VertexId_CUDA) * vertexs, cudaMemcpyHostToDevice, stream);
#else
    printf("CUDA DISABLED Cuda_Stream::move_degree_to_gpu\n");
       exit(0);
#endif
}

void Cuda_Stream::GetWeight(float *edge_weight,    
                            VertexId_CUDA *out_degree,
				VertexId_CUDA *in_degree,
				VertexId_CUDA vertices,
                            VertexId_CUDA *destination,
                            VertexId_CUDA *source,
                            VertexId_CUDA *column_offset,
				VertexId_CUDA *row_indices){
#if CUDA_ENABLE
    uint32_t warp_num = CUDA_NUM_THREADS * CUDA_NUM_BLOCKS/WARP_SIZE;
    uint32_t block_num = std::min(vertices, warp_num);
    get_weight<<<block_num,WARP_SIZE,0,stream>>>(edge_weight,out_degree,in_degree,vertices,destination,source,column_offset,row_indices);
//    this->CUDA_DEVICE_SYNCHRONIZE();
//    print_cuda_sum(edge_weight, vertices, "gpu edge weight");
#else
       printf("CUDA DISABLED Cuda_Stream::get_weight\n");
       exit(0);   
#endif   
}

void Cuda_Stream::GetMeanWeight(float *edge_weight,
                            VertexId_CUDA *out_degree,
                            VertexId_CUDA *in_degree,
                            VertexId_CUDA vertices,
                            VertexId_CUDA *destination,
                            VertexId_CUDA *source,
                            VertexId_CUDA *column_offset,
                            VertexId_CUDA *row_indices){
#if CUDA_ENABLE
    uint32_t warp_num = CUDA_NUM_THREADS * CUDA_NUM_BLOCKS/WARP_SIZE;
    uint32_t block_num = std::min(vertices, warp_num);
    get_mean_weight<<<block_num,WARP_SIZE,0,stream>>>(edge_weight,out_degree,in_degree,vertices,destination,source,column_offset,row_indices);
//    this->CUDA_DEVICE_SYNCHRONIZE();
//    print_cuda_sum(edge_weight, vertices, "gpu edge weight");
#else
    printf("CUDA DISABLED Cuda_Stream::get_weight\n");
       exit(0);
#endif
}

void Cuda_Stream::Edge_Softmax_Forward_Block(float* msg_output,float* msg_input,//data 
        float* msg_cached,
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size){
#if CUDA_ENABLE 
        edge_softmax_forward_block<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS_SOFTMAX,CUDA_NUM_THREADS_SOFTMAX,0,stream>>>(
            msg_output, msg_input, msg_cached, row_indices, column_offset,
            batch_size, feature_size); 
#else
       printf("CUDA DISABLED Cuda_Stream::Edge_Softmax_Forward_Block\n");
       exit(0);   
#endif      
}

void Cuda_Stream::Edge_Softmax_Forward_Norm_Block(float* msg_output,float* msg_input,//data
                                             float* msg_cached,
                                             VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
                                             VertexId_CUDA batch_size, VertexId_CUDA feature_size){
#if CUDA_ENABLE
    float* node_max;
    allocate_gpu_buffer_async(&node_max, batch_size, stream);
    // TODO: node max 有问题
    get_node_max<float, VertexId_CUDA><<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS, 0, stream>>>
            (msg_input, column_offset, batch_size, node_max);
    edge_softmax_forward_norm_block<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS_SOFTMAX,CUDA_NUM_THREADS_SOFTMAX,0,stream>>>(
            msg_output, msg_input, msg_cached, row_indices, column_offset,
            batch_size, feature_size, node_max);
    free_gpu_mem_async(node_max, stream);
#else
    printf("CUDA DISABLED Cuda_Stream::Edge_Softmax_Forward_Block\n");
       exit(0);
#endif
}

//void Cuda_Stream::Get_Src_Max(float* msg_output,float* msg_input,//data
//                                             VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
//                                             VertexId_CUDA batch_size, VertexId_CUDA feature_size){
//#if CUDA_ENABLE
//    edge_softmax_forward_block<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS_SOFTMAX,CUDA_NUM_THREADS_SOFTMAX,0,stream>>>(
//            msg_output, msg_input, msg_cached, row_indices, column_offset,
//            batch_size, feature_size);
//#else
//    printf("CUDA DISABLED Cuda_Stream::Edge_Softmax_Forward_Block\n");
//       exit(0);
//#endif
//}

void Cuda_Stream::Edge_Softmax_Backward_Block(float* msg_input_grad,float* msg_output_grad,//data 
        float* msg_cached,
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size){
#if CUDA_ENABLE
	//printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tfeature_size:%d\n",BLOCK_SIZE,feature_size); 
        edge_softmax_backward_block<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS_SOFTMAX,CUDA_NUM_THREADS_SOFTMAX,0,stream>>>(
            msg_input_grad, msg_output_grad, msg_cached, row_indices, column_offset,
            batch_size, feature_size); 
#else
       printf("CUDA DISABLED Cuda_Stream::Edge_Softmax_Backward_Block\n");
       exit(0);   
#endif      
}



















void move_result_out(float* output,float* input, int src,int dst, int feature_size, bool sync){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpy(output,input,((long)(dst-src))*feature_size*(sizeof(int)), cudaMemcpyDeviceToHost));
    if(sync)
    cudaDeviceSynchronize();
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Message\n");
       exit(0);   
#endif 


}

void move_data_in(float* d_pointer,float* h_pointer, int start, int end, int feature_size, bool sync){
#if CUDA_ENABLE    
    CHECK_CUDA_RESULT(cudaMemcpy(d_pointer,h_pointer,((long)(end-start))*feature_size*(sizeof(float)), cudaMemcpyHostToDevice));
    if(sync)
    cudaDeviceSynchronize();
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Message\n");
       exit(0);   
#endif 
}

void move_edge_in(VertexId_CUDA * d_pointer,VertexId_CUDA* h_pointer, VertexId_CUDA start, VertexId_CUDA end, int feature_size, bool sync){
#if CUDA_ENABLE    
    CHECK_CUDA_RESULT(cudaMemcpy(d_pointer,h_pointer,((long)(end-start))*feature_size*(sizeof(VertexId_CUDA)), cudaMemcpyHostToDevice));
    if(sync)
    cudaDeviceSynchronize();
#else
       printf("CUDA DISABLED move_edge_in\n");
       exit(0);   
#endif 
}

/**
 * 将数据从主机移动到设备端
*/
void move_bytes_in(void * d_pointer,void* h_pointer, long bytes, bool sync){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpy(d_pointer,h_pointer,bytes, cudaMemcpyHostToDevice));
    if(sync)
    cudaDeviceSynchronize();
#else
       printf("CUDA DISABLED move_bytes_in\n");
       exit(0);   
#endif 
}

void move_bytes_in_async_check(void * d_pointer,void* h_pointer, long bytes, cudaStream_t cs){
#if CUDA_ENABLE
    print_cpu_sum((VertexId_CUDA*)h_pointer, bytes/sizeof(VertexId_CUDA), "cpu_destiantion");
    print_cuda_sum<VertexId_CUDA>((VertexId_CUDA*)d_pointer, bytes/sizeof(VertexId_CUDA), "dev destination");
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        // 处理错误
    }
    CHECK_CUDA_RESULT(cudaMemcpy(d_pointer, h_pointer, bytes, cudaMemcpyHostToDevice));
//    CHECK_CUDA_RESULT(cudaMemcpyAsync(d_pointer,h_pointer,bytes, cudaMemcpyHostToDevice, nullptr));
    cudaDeviceSynchronize();
#else
    printf("CUDA DISABLED move_bytes_in\n");
       exit(0);
#endif
}

void move_bytes_in_async(void * d_pointer,void* h_pointer, long bytes, cudaStream_t cs){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpyAsync(d_pointer,h_pointer,bytes, cudaMemcpyHostToDevice, cs));
#else
       printf("CUDA DISABLED move_bytes_in\n");
       exit(0);   
#endif 
}


void move_bytes_out(void * h_pointer,void* d_pointer, long bytes, bool sync){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpy(h_pointer,d_pointer,bytes, cudaMemcpyDeviceToHost));
    if(sync)
    cudaDeviceSynchronize();
#else
       printf("CUDA DISABLED move_bytes_in\n");
       exit(0);   
#endif 
}

void move_bytes_out_async(void * h_pointer,void* d_pointer, long bytes, cudaStream_t cs){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpyAsync(h_pointer,d_pointer,bytes, cudaMemcpyDeviceToHost, cs));
#else
       printf("CUDA DISABLED move_bytes_in\n");
       exit(0);   
#endif 
}

//void aggregate_comm_result(float* aggregate_buffer,float *input_buffer,int data_size,int feature_size,int partition_offset,bool sync){
//#if CUDA_ENABLE
//    const int THREAD_SIZE=512;//getThreadNum(_meta->get_feature_size());
//    const int BLOCK_SIZE=32;
//    aggregate_data_buffer<<<THREAD_SIZE,BLOCK_SIZE>>>(aggregate_buffer,input_buffer,data_size,feature_size,partition_offset,sync);
//    if(sync)
//    	cudaDeviceSynchronize();
//#else
//       printf("CUDA DISABLED aggregate_comm_result\n");
//       exit(0);   
//#endif 
//
//}

void ntsFreeHost(void *buffer){
#if CUDA_ENABLE    
    cudaFreeHost(buffer);
#else
       printf("CUDA DISABLED FreeBuffer\n");
       exit(0);   
#endif 
}


void FreeBuffer(float *buffer){
#if CUDA_ENABLE    
    cudaFree(buffer);
#else
       printf("CUDA DISABLED FreeBuffer\n");
       exit(0);   
#endif 
}

void FreeEdge(VertexId_CUDA *buffer){
#if CUDA_ENABLE
     cudaFree(buffer);
#else
       printf("CUDA DISABLED FreeEdge\n");
       exit(0);   
#endif 
}

void FreeBufferAsync(float *buffer, cudaStream_t cs){
#if CUDA_ENABLE    
    cudaFreeAsync(buffer, cs);
#else
       printf("CUDA DISABLED FreeBuffer\n");
       exit(0);   
#endif 
}

void FreeEdgeAsync(VertexId_CUDA *buffer, cudaStream_t cs){
#if CUDA_ENABLE
     cudaFreeAsync(buffer, cs);
#else
       printf("CUDA DISABLED FreeEdge\n");
       exit(0);   
#endif 
}
void zero_buffer(float* buffer,int size){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemset(buffer,0,sizeof(float)*size));
    cudaDeviceSynchronize();
#else
       printf("CUDA DISABLED zero_buffer\n");
       exit(0);   
#endif 
}


void allocate_gpu_buffer(float** input, int size){
#if CUDA_ENABLE
        CHECK_CUDA_RESULT(cudaMalloc(input,sizeof(float)*(size)));
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Message\n");
       exit(0);   
#endif 

}
void allocate_gpu_edge(VertexId_CUDA** input, int size){
#if CUDA_ENABLE
     CHECK_CUDA_RESULT(cudaMalloc(input,sizeof(VertexId_CUDA)*(size)));
#else 
     printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Message\n");
     exit(0);   
   
#endif 
}

void allocate_gpu_buffer_async(float** input, int size, cudaStream_t cs){
#if CUDA_ENABLE
       // std::printf("stream: %p\n", cs);
        CHECK_CUDA_RESULT(cudaMallocAsync(input,sizeof(float)*(size), cs));
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Message\n");
       exit(0);   
#endif 

}

void free_gpu_mem_async(void* mem, cudaStream_t cs) {
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaFreeAsync(mem, cs));
#else
    printf("CUDA DISABLED free_gpu_mem_async\n");
     exit(0);

#endif
}

void allocate_gpu_edge_async(VertexId_CUDA** input, int size, cudaStream_t cs){
#if CUDA_ENABLE
     CHECK_CUDA_RESULT(cudaMallocAsync(input,sizeof(VertexId_CUDA)*(size), cs));
#else 
     printf("CUDA DISABLED Cuda_Stream::allocate_gpu_edge_async\n");
     exit(0);   
   
#endif 
}

template<typename T>
void sort_graph_vertex(T* vertex_in, T*vertex_out, VertexId_CUDA vertex_num, VertexId_CUDA out_num){
    out_num = std::min(vertex_num, out_num);
    T  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
    T  *d_keys_out;        // e.g., [        ...        ]
    cudaMalloc((void**)&d_keys_in, sizeof(T)*vertex_num);
    cudaMalloc((void**)&d_keys_out, sizeof(T)*vertex_num);
    cudaMemcpy((void*)d_keys_in, (void*)vertex_in, sizeof(T)*vertex_num, cudaMemcpyHostToDevice);
// Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, vertex_num);
// Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, vertex_num);
    cudaMemcpy(&(vertex_out[vertex_num - out_num]), (void*)d_keys_out, sizeof(T)*out_num, cudaMemcpyDeviceToHost);
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_temp_storage);
}

//template<class DevType>
//void free_gpu_memory(DevType* dev_ptr) {
//#if CUDA_ENABLE
//    CHECK_CUDA_RESULT(cudaFree(dev_ptr));
//#else
//    printf("CUDA DISABLED free_gpu_memory\n");
//    exit(0);
//#endif
//}
