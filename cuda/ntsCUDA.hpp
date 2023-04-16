/*
 * test.h
 *
 *  Created on: Dec 3, 2019
 *      Author: wangqg
 */

#include "cuda_type.h"
#define CUDA_ENABLE 1
#if CUDA_ENABLE
#include "cuda_runtime.h"
#include "cusparse.h"
#include "cublas_v2.h"
#endif

#ifndef TEST_HPP
#define TEST_HPP
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <vector>
//#include"../core/graph.hpp"

enum graph_type { CSR, CSC, PAIR };
enum weight_type { NULL_TYPE, SCALA_TYPE, TENSOR_TYPE };

void ntsFreeHost(void *buffer);
void *cudaMallocPinned(long size_of_bytes);
void *getDevicePointer(void *host_data_to_device);
void *cudaMallocGPU(long size_of_bytes);
void move_result_out(float *output, float *input, int src, int dst,
                     int feature_size, bool sync = true);
void move_data_in(float *d_pointer, float *h_pointer, int start, int end,
                  int feature_size, bool sync = true);
void move_edge_in(VertexId_CUDA *d_pointer, VertexId_CUDA *h_pointer,
                  VertexId_CUDA start, VertexId_CUDA end, int feature_size,
                  bool sync = true);
void move_bytes_in(void *d_pointer, void *h_pointer, long bytes,
                   bool sync = true);
void move_bytes_in_async(void * d_pointer,void* h_pointer, long bytes, cudaStream_t cs);
void move_bytes_out(void *h_pointer, void *d_pointer, long bytes,
                   bool sync = true);
void move_bytes_out_async(void * h_pointer,void* d_pointer, long bytes, cudaStream_t cs);
void allocate_gpu_buffer(float **input, int size);
void allocate_gpu_edge(VertexId_CUDA **input, int size);

void allocate_gpu_buffer_async(float **input, int size, cudaStream_t cs);
void allocate_gpu_edge_async(VertexId_CUDA **input, int size, cudaStream_t cs);
template<typename T>
void sort_graph_vertex(T* vertex_in, T*vertex_out, VertexId_CUDA vertex_num, VertexId_CUDA out_num);
void aggregate_comm_result(float *aggregate_buffer, float *input_buffer,
                           int data_size, int feature_size,
                           int partition_offset, bool sync = true);
void FreeBuffer(float *buffer);
void FreeEdge(VertexId_CUDA *buffer);

void FreeBufferAsync(float *buffer, cudaStream_t cs);
void FreeEdgeAsync(VertexId_CUDA *buffer, cudaStream_t cs);

void zero_buffer(float *buffer, int size);
void CUDA_DEVICE_SYNCHRONIZE();
void ResetDevice();

class deviceCSC{
public:
VertexId_CUDA* column_offset;
VertexId_CUDA* row_indices;
VertexId_CUDA* mirror_index;
VertexId_CUDA v_size;
VertexId_CUDA e_size;
VertexId_CUDA mirror_size;
bool require_mirror=false;

deviceCSC(){
    column_offset=NULL;
    row_indices=NULL;
}
void init(VertexId_CUDA v_size_, VertexId_CUDA e_size_,
        bool require_mirror_=false,VertexId_CUDA mirror_size_=0){
    v_size=v_size_;
    e_size=e_size_;
    require_mirror=false;
    column_offset=(VertexId_CUDA*)cudaMallocGPU((v_size_+1)*sizeof(VertexId_CUDA));
    row_indices=(VertexId_CUDA*)cudaMallocGPU((e_size_)*sizeof(VertexId_CUDA));
    if(require_mirror_){
        require_mirror=require_mirror_;
        mirror_size=mirror_size_;
        mirror_index=(VertexId_CUDA*)cudaMallocGPU((mirror_size_)*sizeof(VertexId_CUDA));
    }
}
void load_from_host(VertexId_CUDA* h_column_offset,VertexId_CUDA* h_row_indices,
            VertexId_CUDA* h_mirror_index){
   // printf("%d %d %d \n",v_size,e_size,mirror_size);
    move_bytes_in(column_offset,h_column_offset,(v_size+1)*sizeof(VertexId_CUDA));
    move_bytes_in(row_indices,h_row_indices,(e_size)*sizeof(VertexId_CUDA));
    move_bytes_in(mirror_index,h_mirror_index,(mirror_size)*sizeof(VertexId_CUDA));
}
void load_from_host(VertexId_CUDA* h_column_offset,VertexId_CUDA* h_row_indices){
    move_bytes_in(column_offset,h_column_offset,(v_size+1)*sizeof(VertexId_CUDA));
    move_bytes_in(row_indices,h_row_indices,(e_size)*sizeof(VertexId_CUDA));
}
void release(){
    FreeEdge(column_offset);
    FreeEdge(row_indices);
    if(require_mirror)
        FreeEdge(mirror_index);
}
~deviceCSC(){
}
};

class Cuda_Stream {
public:
    // toao debug
    double cpu_inclusiveTime = 0.0;
  double inclusiveTime = 0;

  Cuda_Stream();
  void destory_Stream();
  cudaStream_t getStream();
  void setNewStream(cudaStream_t cudaStream);
  void CUDA_DEVICE_SYNCHRONIZE();
    void CUDA_SYNCHRONIZE_ALL();
  cudaStream_t stream;
  cusparseHandle_t sparse_handle = NULL;
  cublasHandle_t blas_handle;
  void* cuda_buffer = NULL;
  size_t cuda_buffer_size = 0;
  unsigned char* cpu_buffer = NULL;
  size_t cpu_buffer_size = 0;

  void move_result_out(float *output, float *input, VertexId_CUDA src,
                       VertexId_CUDA dst, int feature_size, bool sync = true);
  void move_data_in(float *d_pointer, float *h_pointer, VertexId_CUDA start,
                    VertexId_CUDA end, int feature_size, bool sync = true);
  void move_edge_in(VertexId_CUDA *d_pointer, VertexId_CUDA *h_pointer,
                    VertexId_CUDA start, VertexId_CUDA end, int feature_size,
                    bool sync = true);
  void aggregate_comm_result(float *aggregate_buffer, float *input_buffer,
                             VertexId_CUDA data_size, int feature_size,
                             int partition_offset, bool sync = true);
  void deSerializeToGPU(float *input_gpu_buffer, float *input_buffer,
                        VertexId_CUDA data_size, VertexId_CUDA feature_size,
                        VertexId_CUDA partition_start,
                        VertexId_CUDA partition_end, bool sync);
  void aggregate_comm_result_debug(float *aggregate_buffer, float *input_buffer,
                                   VertexId_CUDA data_size,
                                   VertexId_CUDA feature_size,
                                   VertexId_CUDA partition_start,
                                   VertexId_CUDA partition_end, bool sync);
  
//fused op
  void Gather_By_Dst_From_Src(
      float *input, float *output, float *weight_forward,       // data
      VertexId_CUDA *row_indices, VertexId_CUDA *column_offset, // graph
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end,VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);

  void Gather_By_Dst_From_Src_Spmm(
      float *input, float *output, float *weight_forward,       // data
      VertexId_CUDA *row_indices, VertexId_CUDA *column_offset, VertexId_CUDA column_num, // graph
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end,VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);

  void Push_From_Dst_To_Src(
      float *input, float *output, float *weight_forward,       // data
      VertexId_CUDA *row_indices, VertexId_CUDA *column_offset, // graph
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end,VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);
    void Push_From_Dst_To_Src_Spmm(
            float *input, float *output, float *weight_forward,       // data
            VertexId_CUDA *row_indices, VertexId_CUDA *column_offset, int column_num, // graph
            VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
            VertexId_CUDA dst_end,VertexId_CUDA edges, VertexId_CUDA batch_size,
            VertexId_CUDA feature_size, bool with_weight = false,
            bool tensor_weight = false);
  void Gather_By_Dst_From_Src_with_cache(
      float *input, float *output, float *weight_forward,       // data
      VertexId_CUDA *cacheflag, VertexId_CUDA *destination,
      VertexId_CUDA *row_indices, VertexId_CUDA *column_offset, // graph
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end,VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);
  void Gather_By_Dst_From_Src_Optim(
      float *input, float *output, float *weight_forward, // data
      VertexId_CUDA *row_indices, VertexId_CUDA *column_offset,
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end,VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);
  void Gather_By_Src_From_Dst_Optim(
      float *input, float *output, float *weight_forward, // data
      VertexId_CUDA *row_offset, VertexId_CUDA *column_indices,
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);
  void Gather_By_Src_From_Dst(
      float *input, float *output, float *weight_forward,       // data
      VertexId_CUDA *row_offset, VertexId_CUDA *column_indices, // graph
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);
  void Gather_By_Src_From_Dst_Spmm(
      float *input, float *output, float *weight_forward,       // data
      VertexId_CUDA *row_offset, VertexId_CUDA *column_indices, VertexId_CUDA column_num, // graph
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);
  
  void Scatter_Src_Mirror_to_Msg(float* message,float* src_mirror_feature,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
        VertexId_CUDA* mirror_index, VertexId_CUDA batch_size,
        VertexId_CUDA feature_size);

  void Gather_Msg_To_Src_Mirror(float* src_mirror_feature,float* message,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
        VertexId_CUDA* mirror_index, VertexId_CUDA batch_size,
        VertexId_CUDA feature_size);

  void Scatter_Dst_to_Msg(float* message,float* dst_feature,//data 
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size);

  void Gather_Msg_to_Dst(float* dst_feature,float* message,//data 
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size);

  void Edge_Softmax_Forward_Block(float* msg_output,float* msg_input,//data 
        float* msg_cached,
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size);

  void Edge_Softmax_Backward_Block(float* msg_input_grad,float* msg_output_grad,//data 
        float* msg_cached,
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size);

  void sample_processing_get_co_gpu(VertexId_CUDA *dst, VertexId_CUDA *local_column_offset,
                                   VertexId_CUDA *global_column_offset,
                                   VertexId_CUDA dst_size, 
                                   VertexId_CUDA* tmp_data_buffer,
                                   VertexId_CUDA src_index_size,
					               VertexId_CUDA* src_count,
					               VertexId_CUDA* src_index,
                                   VertexId_CUDA fanout,
                                   VertexId_CUDA& edge_size);

  void sample_processing_get_co_gpu_omit(VertexId_CUDA* CacheFlag, VertexId_CUDA *dst, VertexId_CUDA *local_column_offset,
                                   VertexId_CUDA *global_column_offset,
                                   VertexId_CUDA dst_size, 
                                   VertexId_CUDA* tmp_data_buffer,
                                   VertexId_CUDA src_index_size,
					               VertexId_CUDA* src_count,
					               VertexId_CUDA* src_index,
                                   VertexId_CUDA fanout,
                                   VertexId_CUDA& edge_size);

  void sample_processing_update_ri_gpu(VertexId_CUDA *r_i,
								 	VertexId_CUDA *src_index,
                                   	VertexId_CUDA edge_size,
                                    VertexId_CUDA src_index_size);
  void sample_processing_traverse_gpu(VertexId_CUDA *destination,
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
                                      VertexId_CUDA max_sample_num);

  void zero_copy_feature_move_gpu(float *dev_feature,
						float *pinned_host_feature,
						VertexId_CUDA *src_vertex,
                        VertexId_CUDA feature_size,
						VertexId_CUDA vertex_size);
  void global_copy_label_move_gpu(long *dev_label,
                    long *global_dev_label,
                    VertexId_CUDA *dst_vertex,
                    VertexId_CUDA vertex_size);
  void dev_Grad_refresh(float *dev_grad_buffer,
				float *dev_share_grad,
				VertexId_CUDA *dev_cacheflag,
                VertexId_CUDA *dev_cachemap,
                VertexId_CUDA feature_size,
                VertexId_CUDA *destination_vertex,
				VertexId_CUDA vertex_size);
  void dev_Grad_accumulate(float *dev_grad_buffer,
				float *dev_share_grad,
				VertexId_CUDA *dev_cacheflag,
                VertexId_CUDA *dev_cachemap,
                VertexId_CUDA feature_size,
                VertexId_CUDA *destination_vertex,
				VertexId_CUDA vertex_size);
  void dev_load_share_embedding(float *dev_embedding,
                    float *share_embedding,
                    VertexId_CUDA *dev_cacheflag,
                    VertexId_CUDA *dev_cachemap,
                    VertexId_CUDA feature_size,
                    VertexId_CUDA *destination_vertex,
				    VertexId_CUDA vertex_size);
  void dev_update_share_embedding(float *dev_embedding,
                    float *share_embedding,
                    VertexId_CUDA *dev_cachemap,
                    VertexId_CUDA *dev_cacheflag,
                    VertexId_CUDA feature_size,
                    VertexId_CUDA *destination_vertex,
				    VertexId_CUDA vertex_size);
  void zero_copy_embedding_move_gpu(float *dev_feature,
						float *pinned_host_feature,
                        VertexId_CUDA feature_size,
						VertexId_CUDA vertex_size);
  void ReFreshDegree(VertexId_CUDA *out_degree,
				                  VertexId_CUDA *in_degree,
				                  VertexId_CUDA vertices);
  void UpdateDegree(VertexId_CUDA *out_degree,
				                 VertexId_CUDA *in_degree,
				                 VertexId_CUDA vertices,
                                 VertexId_CUDA *destination,
                                 VertexId_CUDA *source,
                                 VertexId_CUDA *column_offset,
				                 VertexId_CUDA *row_indices);
  void GetWeight(float *edge_weight,
                 VertexId_CUDA *out_degree,
				 VertexId_CUDA *in_degree,
				 VertexId_CUDA vertices,
                 VertexId_CUDA *destination,
                 VertexId_CUDA *source,
                 VertexId_CUDA *column_offset,
				 VertexId_CUDA *row_indices);
  
  
  void Gather_By_Dst_From_Message(
      float *input, float *output,            // data
      VertexId_CUDA *src, VertexId_CUDA *dst, // graph
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);
  void Scatter_Grad_Back_To_Message(
      float *input, float *message_grad, // data
      VertexId_CUDA *row_indices, VertexId_CUDA *column_offset,
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = true);
//  void process_local(float *local_buffer, float *input_tensor,
//                     VertexId_CUDA *src, VertexId_CUDA *dst,
//                     VertexId_CUDA *src_index, float *weight_buffer,
//                     int dst_offset, int dst_offset_end, int feature_size,
//                     int edge_size, bool sync = true);
//  void process_local_inter(float *local_buffer, float *input_tensor,
//                           VertexId_CUDA *src, VertexId_CUDA *dst,
//                           VertexId_CUDA *src_index, VertexId_CUDA *dst_index,
//                           float *weight_buffer, int dst_offset,
//                           int dst_offset_end, int feature_size, int edge_size,
//                           int out_put_buffer_size, bool sync = true);
//  void process_local_inter_para(float *local_buffer, float *input_tensor,
//                                VertexId_CUDA *src, VertexId_CUDA *dst,
//                                VertexId_CUDA *src_index,
//                                VertexId_CUDA *dst_index, float *para,
//                                int dst_offset, int dst_offset_end,
//                                int feature_size, int edge_size,
//                                int out_put_buffer_size, bool sync = true);


    void dev_load_share_embedding(float *dev_embedding, float *share_embedding, VertexId_CUDA *dev_cacheflag,
                                  VertexId_CUDA *dev_cachemap, VertexId_CUDA feature_size,
                                  VertexId_CUDA *destination_vertex, uint8_t *dev_x_mask, uint8_t *dev_cache_mask,
                                  VertexId_CUDA vertex_size);


    void dev_update_share_embedding_and_feature(float *dev_feature, float *dev_embedding, float *share_aggregate,
                                                float *share_embedding, VertexId_CUDA *dev_cachemap,
                                                VertexId_CUDA *dev_cacheflag, VertexId_CUDA feature_size,
                                                VertexId_CUDA embedding_size, VertexId_CUDA *destination_vertex,
                                                VertexId_CUDA *dev_X_version, VertexId_CUDA *dev_Y_version,
                                                VertexId_CUDA vertex_size, VertexId_CUDA require_version);

    void dev_load_share_embedding_and_feature(float *dev_feature, float *dev_embedding, float *share_feature,
                                              float *share_embedding, VertexId_CUDA *dev_cacheflag,
                                              VertexId_CUDA *dev_cachemap, VertexId_CUDA feature_size,
                                              VertexId_CUDA embedding_size, VertexId_CUDA *destination_vertex,
                                              VertexId_CUDA vertex_size);

    void move_degree_to_gpu(VertexId_CUDA *cpu_in_degree, VertexId_CUDA *cpu_out_degree, VertexId_CUDA *gpu_in_degree,
                            VertexId_CUDA *gpu_out_degree, VertexId_CUDA vertexs);

    void dev_load_share_aggregate(float *dev_feature, float *share_feature, VertexId_CUDA *dev_cacheflag,
                                  VertexId_CUDA *dev_cachemap, VertexId_CUDA feature_size,
                                  VertexId_CUDA *destination_vertex, VertexId_CUDA vertex_size);
};

// int test();
#endif /* TEST_H_ */
