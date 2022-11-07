
#include"cuda_type.h"
#include "ntsCUDA.hpp"

#if CUDA_ENABLE
#include "ntsCUDAFuseKernel.cuh"
#include "ntsCUDADistKernel.cuh"
#include "ntsCUDATransferKernel.cuh"

#endif

#if CUDA_ENABLE
#define CHECK_CUDA_RESULT(N) {											\
	cudaError_t result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }
#endif


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


Cuda_Stream::Cuda_Stream(){
#if CUDA_ENABLE
       CHECK_CUDA_RESULT(cudaStreamCreate(&stream));
#else
       printf("CUDA DISABLED Cuda_Stream::Cuda_Stream\n");
       exit(0);  
#endif  
}

void Cuda_Stream::destory_Stream(){
#if CUDA_ENABLE
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
       cudaStreamSynchronize(stream);
#else
       printf("CUDA DISABLED Cuda_Stream::CUDA_DEVICE_SYNCHRONIZE\n");
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

void Cuda_Stream::sample_processing_get_co_gpu(VertexId_CUDA *dst, 
                                   VertexId_CUDA *local_column_offset,
                                   VertexId_CUDA *global_column_offset,
                                   VertexId_CUDA dst_size,
                                   VertexId_CUDA* tmp_data_buffer,
                                   VertexId_CUDA src_index_size,
					VertexId_CUDA* src_count,
					VertexId_CUDA* src_index){
#if CUDA_ENABLE
    sample_processing_get_co_gpu_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                            (dst,tmp_data_buffer,global_column_offset,dst_size,
                            src_index_size,src_count,src_index);
    this->CUDA_DEVICE_SYNCHRONIZE();
    int num_items = dst_size + 1;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, tmp_data_buffer, local_column_offset, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    //printf("temp_storage_bytes:%d num_items:%d\n",temp_storage_bytes,num_items);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, tmp_data_buffer, local_column_offset, num_items);
#else
       printf("CUDA DISABLED Cuda_Stream::sample_processing_get_co_gpu\n");
       exit(0);   
#endif     
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
						       VertexId_CUDA* src_count){
#if CUDA_ENABLE
    sample_processing_traverse_gpu_kernel_stage2<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                            (destination, c_o,r_i,global_c_o,global_r_i,src_index,vtx_size);
    this->CUDA_DEVICE_SYNCHRONIZE();
    sample_processing_traverse_gpu_kernel_stage3<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                            (src_index,src_index_size,src,src_count);
    this->CUDA_DEVICE_SYNCHRONIZE();
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
    this->CUDA_DEVICE_SYNCHRONIZE();
#else
    printf("CUDA DISABLED Cuda_Stream::sample_processing_update_ri_gpu\n"); 
    exit(0);   
#endif   
}

void Cuda_Stream::zero_copy_feature_move_gpu(float *dev_feature,
						float *pinned_host_feature,
						VertexId_CUDA *src_vertex,
                                   	VertexId_CUDA feature_size,
						VertexId_CUDA vertex_size){
#if CUDA_ENABLE
    zero_copy_feature_move_gpu_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                            (dev_feature,pinned_host_feature,src_vertex,feature_size,vertex_size);
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
    global_copy_label_move_gpu_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                            (dev_label,global_dev_label,dst_vertex,vertex_size);
    this->CUDA_DEVICE_SYNCHRONIZE();
#else
       printf("CUDA DISABLED Cuda_Stream::copy_label_from_global\n");
       exit(0);   
#endif   
}

void Cuda_Stream::dev_updata_share_embedding(float *dev_embedding,
				float *share_embedding,
				VertexId_CUDA *dev_cacheflag,
                            VertexId_CUDA *dev_cacheepoch,
                            VertexId_CUDA current_epoch,
                            VertexId_CUDA feature_size,
                            VertexId_CUDA *destination_vertex,
				VertexId_CUDA vertex_size){
#if CUDA_ENABLE
    dev_updata_share_embedding_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                            (dev_embedding,share_embedding,dev_cacheflag,dev_cacheepoch, current_epoch, feature_size,destination_vertex,vertex_size);
    this->CUDA_DEVICE_SYNCHRONIZE();
#else
       printf("CUDA DISABLED Cuda_Stream::copy_label_from_global\n");
       exit(0);   
#endif   
}

void Cuda_Stream::dev_load_share_embedding(float *dev_embedding,
				float *share_embedding,
				VertexId_CUDA *dev_cacheflag,
                            VertexId_CUDA feature_size,
                            VertexId_CUDA *destination_vertex,
				VertexId_CUDA vertex_size){
#if CUDA_ENABLE
    dev_updata_load_embedding_kernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>
                            (dev_embedding,share_embedding,dev_cacheflag,feature_size,destination_vertex,vertex_size);
    this->CUDA_DEVICE_SYNCHRONIZE();
#else
       printf("CUDA DISABLED Cuda_Stream::copy_label_from_global\n");
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
    this->CUDA_DEVICE_SYNCHRONIZE();
#else
       printf("CUDA DISABLED Cuda_Stream::up_date_degree\n");
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
    get_weight<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(edge_weight,out_degree,in_degree,vertices,destination,source,column_offset,row_indices);
    this->CUDA_DEVICE_SYNCHRONIZE();
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
