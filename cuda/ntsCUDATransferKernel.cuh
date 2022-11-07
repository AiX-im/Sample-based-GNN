/*
 * propagate.h
 *
 *  Created on: Dec 3, 2019
 *      Author: wangqg
 * TODO :cub support and shared memory optimization
 */

#ifndef NTSCUDATRANSFERKERNEL_CUH
#define NTSCUDATRANSFERKERNEL_CUH

#include"cuda_type.h"
#include<stdlib.h>
#include<stdio.h>
#include<cstdio>
#include<assert.h>
#include <thrust/extrema.h>
#include <sys/time.h>
#include<cuda.h>
#include"cub/cub.cuh"
#include"math.h"
inline double get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (tv.tv_usec / 1e6);
  }



//util for fused graph op
__global__ void aggregate_data_buffer(float *result_buffer,float *comm_buffer,
 		size_t data_size,int feature_size,int partition_offset,bool debug=false){
			
	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
     size_t record_size=feature_size+1;//with key attached;
	for(long i=threadId;i<(long)feature_size*data_size;i+=blockDim.x*gridDim.x){
            size_t rank=i%(feature_size);
            unsigned int key=i/(feature_size);
            unsigned int *v_id=NULL;
            unsigned int id=0;
            v_id=(unsigned int*)(comm_buffer+(key*record_size));
			atomicAdd(&comm_buffer[key*record_size+rank+1],comm_buffer[key*record_size+rank+1]);
			//atomicAdd(&result_buffer[feature_size*((*v_id)-partition_offset)+rank],result_buffer[feature_size*((*v_id)-partition_offset)+rank]);
			//atomicAdd(&result_buffer[feature_size*((*v_id)-partition_offset)+rank],comm_buffer[key*record_size+rank+1]);
		
	}
	if(threadId==0)printf("partition_offset %d\n",partition_offset);
}

__global__ void aggregate_data_buffer_debug(float *result_buffer,float *comm_buffer,
	size_t data_size,size_t feature_size,size_t partition_start,size_t partition_end,bool debug=false){
	   
size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
size_t record_size=feature_size+1;//with key attached;
for(long i=threadId;i<(long)feature_size*data_size;i+=blockDim.x*gridDim.x){
	   size_t rank=i%(feature_size);
	   long  key=i/(feature_size);
	   unsigned int *v_id=NULL;
	   v_id=(unsigned int*)(comm_buffer+(key*record_size));
           
	   //if((partition_start>(*v_id)||partition_end<=(*v_id))&&i==0)
	   //printf("something wrong %d\n",(*v_id));
	  // atomicAdd(&comm_buffer[key*record_size+rank+1],comm_buffer[key*record_size+rank+1]);
	   //atomicAdd(&result_buffer[feature_size*((*v_id)-partition_start)+rank],result_buffer[feature_size*((*v_id)-partition_offset)+rank]);
	   atomicAdd(&result_buffer[feature_size*((*v_id)-partition_start)+rank],comm_buffer[key*record_size+rank+1]);
   
}
//if(threadId==0)printf("partition_start %d partition_end %d\n",partition_start,partition_end);
}

__global__ void deSerializeToGPUkernel(float *input_gpu_buffer,float *comm_buffer,
	size_t data_size,size_t feature_size,size_t partition_start,size_t partition_end,bool debug=false){
	   
size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
size_t record_size=feature_size+1;//with key attached;
for(long i=threadId;i<(long)feature_size*data_size;i+=blockDim.x*gridDim.x){
	   size_t rank=i%(feature_size);
	   long  key=i/(feature_size);
	   unsigned int *v_id=NULL;
	   v_id=(unsigned int*)(comm_buffer+(key*record_size));
	   //if((partition_start>(*v_id)||partition_end<=(*v_id))&&rank==0)
	   //printf("something wrong1 %d\n",(*v_id));
	   //atomicAdd(&comm_buffer[key*record_size+rank+1],comm_buffer[key*record_size+rank+1]);
	   //atomicAdd(&input_gpu_buffer[feature_size*((*v_id)%10-partition_start)+rank],input_gpu_buffer[feature_size*((*v_id)%10-partition_start)+rank]);
           if((partition_start<=(*v_id)&&partition_end>(*v_id))){
              // if((*v_id)==875712&&rank==0)printf("data %d %f %d,\n",(*v_id), comm_buffer[key*record_size+rank+1],partition_end);
           //atomicAdd(&input_gpu_buffer[feature_size*((*v_id)-partition_start)+rank],input_gpu_buffer[feature_size*((*v_id)-partition_start)+rank]);
               input_gpu_buffer[feature_size*((*v_id)-partition_start)+rank]=comm_buffer[key*record_size+rank+1];
           }
	   //atomicAdd(&input_gpu_buffer[feature_size*((*v_id)-partition_start)+rank],comm_buffer[key*record_size+rank+1]);
   
}
//if(threadId==0)printf("partition_start %d partition_end %d\n",partition_start,partition_end);
}

__global__ void zero_copy_feature_move_gpu_kernel(float *dev_feature,
								 	float *pinned_host_feature,
									VertexId_CUDA *src_vertex,
                                   	VertexId_CUDA feature_size,
									VertexId_CUDA vertex_size){
	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	const int WARPSIZE=32;
	size_t laneId =threadId%WARPSIZE;
	size_t warp_id=threadId/WARPSIZE;

	for(long i=threadId;i<(long)vertex_size*WARPSIZE;i+=blockDim.x*gridDim.x){
		VertexId_CUDA vtx_idx=i/WARPSIZE;
		VertexId_CUDA vtx_id=src_vertex[vtx_idx];
		for(int j=laneId;j<feature_size;j+=32){
			dev_feature[vtx_idx*feature_size+j]=
				pinned_host_feature[vtx_id*feature_size+j];
		}
	}
}
__global__ void zero_copy_embedding_move_gpu_kernel(float *dev_feature,
								 	float *pinned_host_feature,
                                   	VertexId_CUDA feature_size,
									VertexId_CUDA vertex_size){
	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	const int WARPSIZE=32;
	size_t laneId =threadId%WARPSIZE;
	size_t warp_id=threadId/WARPSIZE;

	for(long i=threadId;i<(long)vertex_size*WARPSIZE;i+=blockDim.x*gridDim.x){
		VertexId_CUDA vtx_idx=i/WARPSIZE;
		for(int j=laneId;j<feature_size;j+=32){
			dev_feature[vtx_idx*feature_size+j]=
				pinned_host_feature[vtx_idx*feature_size+j];
		}
	}
}
__global__ void global_copy_label_move_gpu_kernel(long *dev_label,
								 	long *global_dev_label,
									VertexId_CUDA *dst_vertex,
									VertexId_CUDA vertex_size){
	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(long i=threadId;i<(long)vertex_size;i+=blockDim.x*gridDim.x){
		VertexId_CUDA dst_vtx = dst_vertex[i];
					  dev_label[i] = global_dev_label[dst_vtx];
	}
}

__global__ void dev_updata_share_embedding_kernel(float *dev_embedding,
								 	float *share_embedding,
									VertexId_CUDA *dev_cacheflag,
									VertexId_CUDA *dev_cacheepoch,
                            		VertexId_CUDA current_epoch,
									VertexId_CUDA feature_size,
									VertexId_CUDA *destination_vertex,
									VertexId_CUDA vertex_size){
	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	const int WARPSIZE=32;
	size_t laneId =threadId%WARPSIZE;
	size_t warp_id=threadId/WARPSIZE;
	
	for(long i=threadId;i<(long)vertex_size*WARPSIZE;i+=blockDim.x*gridDim.x){
		VertexId_CUDA vtx_idx=i/WARPSIZE;
		VertexId_CUDA vtx_id=destination_vertex[vtx_idx];
		if(dev_cacheflag[vtx_id] == 0){
			dev_cacheepoch[vtx_id] = current_epoch;
			for(int j=laneId;j<feature_size;j+=32){
				share_embedding[vtx_id*feature_size+j] = dev_embedding[vtx_idx*feature_size+j];
			}
			dev_cacheflag[vtx_id] = 1;  
		}
	}
}

__global__ void re_fresh_degree(VertexId_CUDA *out_degree,
				                VertexId_CUDA *in_degree,
				                VertexId_CUDA vertices){
	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(long i = threadId; i < (long)vertices; i += blockDim.x*gridDim.x){
		out_degree[i] = 0;
		in_degree[i] = 0;
	}
}

__global__ void up_date_degree(VertexId_CUDA *out_degree,
				   			   VertexId_CUDA *in_degree,
				   			   VertexId_CUDA vertices,
                               VertexId_CUDA *destination,
                               VertexId_CUDA *source,
                               VertexId_CUDA *column_offset,
				   			   VertexId_CUDA *row_indices){
	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	const int WARPSIZE=32;
	size_t laneId =threadId%WARPSIZE;
	size_t warp_id=threadId/WARPSIZE;
	
	for(long i = threadId;i < (long)vertices; i += blockDim.x*gridDim.x){
        long begin_edge = column_offset[i],
             end_edge = column_offset[i + 1];
		VertexId_CUDA dst = destination[i];
        in_degree[dst] = end_edge - begin_edge;

        for (int edge = begin_edge; edge < end_edge; edge++) {
             VertexId_CUDA src = source[row_indices[edge]];
             atomicAdd(&out_degree[src], 1);
        }
	}
}

__global__ void get_weight(float *edge_weight,	 
	 					   VertexId_CUDA *out_degree,
				   		   VertexId_CUDA *in_degree,
				   		   VertexId_CUDA vertices,
                           VertexId_CUDA *destination,
                           VertexId_CUDA *source,
                           VertexId_CUDA *column_offset,
				   		   VertexId_CUDA *row_indices){
	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	const int WARPSIZE = 32;
	size_t laneId = threadId%WARPSIZE;
	size_t warp_id = threadId/WARPSIZE;

	if(warp_id < vertices) {
        const uint64_t start = column_offset[warp_id];
        const uint64_t end = column_offset[warp_id+1];

        for(uint64_t i = start + laneId; i < end; i += 32) {
                long src = row_indices[i];
				edge_weight[i] = 1 / (sqrtf(out_degree[source[src]]) * sqrtf(in_degree[destination[warp_id]]));
        }
    }
}

__global__ void dev_updata_load_embedding_kernel(float *dev_embedding,
								 	float *share_embedding,
									VertexId_CUDA *dev_cacheflag,
									VertexId_CUDA feature_size,
									VertexId_CUDA *destination_vertex,
									VertexId_CUDA vertex_size){
	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	const int WARPSIZE=32;
	size_t laneId =threadId%WARPSIZE;
	size_t warp_id=threadId/WARPSIZE;
	
	for(long i=threadId;i<(long)vertex_size*WARPSIZE;i+=blockDim.x*gridDim.x){
		VertexId_CUDA vtx_idx=i/WARPSIZE;
		VertexId_CUDA vtx_id=destination_vertex[vtx_idx];
		if(dev_cacheflag[vtx_id] == 1){
			for(int j=laneId;j<feature_size;j+=32){
				dev_embedding[vtx_idx*feature_size+j] = share_embedding[vtx_id*feature_size+j];
			}
		}
	}
}
__global__ void sample_processing_get_co_gpu_kernel(VertexId_CUDA *dst,
								 	VertexId_CUDA *local_column_offset,
                                   	VertexId_CUDA *global_column_offset,
                                   	VertexId_CUDA dst_size,
									VertexId_CUDA src_index_size,
									VertexId_CUDA* src_count,
									VertexId_CUDA* src_index
									)
{
	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(long i = threadId; i < (long)dst_size; i += blockDim.x * gridDim.x){
	   	VertexId_CUDA dst_vtx = dst[i];
		local_column_offset[i + 1] = global_column_offset[dst_vtx + 1] - global_column_offset[dst_vtx];
	}
	for(long i = threadId;i < (long)src_index_size;i += blockDim.x*gridDim.x){
       	src_index[i] = 0;
	}
}
__global__ void sample_processing_traverse_gpu_kernel_stage2(
									VertexId_CUDA *destination,
									VertexId_CUDA *c_o,
								 	VertexId_CUDA *r_i,
									VertexId_CUDA *global_c_o,
									VertexId_CUDA *global_r_i,
									VertexId_CUDA *src_index,
                                   	VertexId_CUDA vtx_size){

	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	const int WARPSIZE = 32;
	size_t laneId = threadId % WARPSIZE;
	size_t warp_id = threadId / WARPSIZE;

	for(long i = threadId; i < (long)vtx_size * WARPSIZE; i += blockDim.x*gridDim.x){
       	VertexId_CUDA dst_pos = i / WARPSIZE;
		VertexId_CUDA dst = destination[dst_pos];
	   	VertexId_CUDA e_start_g = global_c_o[dst];
		VertexId_CUDA e_end_g = global_c_o[dst + 1];
		VertexId_CUDA e_start_l = c_o[dst_pos];
		VertexId_CUDA e_end_l = c_o[dst_pos + 1];
		for(long e_id = laneId; e_id < e_end_g - e_start_g; e_id += WARPSIZE){
			VertexId_CUDA src_vtx = global_r_i[e_start_g + e_id];
			r_i[e_start_l + e_id] = src_vtx;
			src_index[src_vtx] = 1;
		}
	}
}

__global__ void sample_processing_traverse_gpu_kernel_stage3(
									VertexId_CUDA *src_index,
									VertexId_CUDA src_index_size,
									VertexId_CUDA* src,
									VertexId_CUDA* src_count){

	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	const int WARPSIZE = 32;
	size_t laneId = threadId % WARPSIZE;
	size_t warp_id = threadId / WARPSIZE;
	for(long i = threadId; i < src_index_size; i += blockDim.x*gridDim.x){
       	if(src_index[i] != 0){
            uint32_t allocation = atomicAdd(src_count, 1); // Just a naive atomic add
			src[allocation] = i;
			src_index[i] = allocation;
		}
	}
}

__global__ void sample_processing_update_ri_gpu_kernel(VertexId_CUDA *r_i,
								 	VertexId_CUDA *src_index,
                                   	VertexId_CUDA edge_size,
									VertexId_CUDA src_index_size){
	   
	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(long i = threadId; i < (long)edge_size; i += blockDim.x*gridDim.x){
	   	VertexId_CUDA src_vtx = r_i[i];
		r_i[i] = src_index[src_vtx];
	}
}














#endif /* PROPAGATE_H_ */
