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

// 一个warp处理一个顶点
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
// 确定每个节点要采的数量，比配置的少则直接为出边数，结果存在第二个参数中
__global__ void sample_processing_get_co_gpu_kernel(VertexId_CUDA *dst,
								 	VertexId_CUDA *local_column_offset,
                                   	VertexId_CUDA *global_column_offset,
                                   	VertexId_CUDA dst_size,
									VertexId_CUDA src_index_size,
									VertexId_CUDA* src_count,
									VertexId_CUDA* src_index,
									VertexId_CUDA fanout
									)
{
	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(long i = threadId; i < (long)dst_size; i += blockDim.x * gridDim.x){
	   	VertexId_CUDA dst_vtx = dst[i];
		local_column_offset[i + 1] = fminf(global_column_offset[dst_vtx + 1] - global_column_offset[dst_vtx], fanout);
		//local_column_offset[i + 1] = global_column_offset[dst_vtx + 1] - global_column_offset[dst_vtx];
	}
}
__global__ void sample_processing_traverse_gpu_kernel_stage2(
									VertexId_CUDA *destination, // 这层要采的节点
									VertexId_CUDA *c_o,         // 采样后的column_offset
								 	VertexId_CUDA *r_i,         // 采样后的row_indices，存储放边的地方
									VertexId_CUDA *global_c_o,  // 存储的是全局的column offset
									VertexId_CUDA *global_r_i,  // 存储的是全局的row_indices
									VertexId_CUDA *src_index,   // 记录节点位于第几层
                                   	VertexId_CUDA vtx_size,     // 节点总数
									VertexId_CUDA layer){       // 当前层数

	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	const int WARPSIZE = 32;
	size_t laneId = threadId % WARPSIZE;
	size_t warp_id = threadId / WARPSIZE;
	for(long i = threadId; i < (long)vtx_size * WARPSIZE; i += blockDim.x*gridDim.x){
       	VertexId_CUDA dst_pos = i / WARPSIZE;           // 使用一个warp来对一个顶点进行采样
		VertexId_CUDA dst = destination[dst_pos];
	   	VertexId_CUDA e_start_g = global_c_o[dst];		// global中该节点的边的开始索引
		VertexId_CUDA e_end_g = global_c_o[dst + 1];	// global中该节点对应的边的结束索引
		VertexId_CUDA e_start_l = c_o[dst_pos];			// 该节点采样的边的起始索引位置
		VertexId_CUDA e_end_l = c_o[dst_pos + 1];		// 该节点采样的边的结束索引位置

		// 下面是并行进行顺序取边
		for(long e_id = laneId; e_id < e_end_l - e_start_l; e_id += WARPSIZE){
            VertexId_CUDA src_vtx = global_r_i[e_start_g + e_id];
			r_i[e_start_l + e_id] = src_vtx;
			src_index[src_vtx] = layer + 1;
		}
	}
}


__device__ __forceinline__ int Log2UpCUDA(int x) {
    if (x <= 2) return x - 1;
    return 32 - __clz(x - 1);
}
class ntsRandom{
    unsigned long long next_random;
public:
    __host__ __device__ ntsRandom(unsigned long long seed) {
        // 这是一个hash算法
        next_random = seed;
        nextValue();
    }
    __host__ __device__ int rand() {
        int ret_value = (int) (next_random);
        nextValue();
        return ret_value;
    }

    __host__ __device__ int rand(int min, int max) {
        assert(max > min);
        int len = max - min;
        return rand() % len + min;
    }
    __host__ __device__ int rand(int max) {
        return rand() % max;
    }

    __host__ __device__ void nextValue() {
        next_random = (next_random * 314159269 + 453806245) & 0x7fffffffULL;
    }
};
template<int BLOCK_DIM=32, int ITEMS_PER_THREAD = 1>
__global__ void sample_processing_traverse_gpu_kernel_stage2(VertexId_CUDA* sample_indices,
                                    VertexId_CUDA* sample_offset,
                                    VertexId_CUDA* destinations,
                                    VertexId_CUDA input_node_count,
                                    VertexId_CUDA * global_column_offset,
                                    VertexId_CUDA* global_row_indices,
                                    VertexId_CUDA* src_index,
                                    VertexId_CUDA max_sample_count,
                                    VertexId_CUDA layer,
                                    unsigned long long random_seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;    // 每个blockIdx.x负责一个顶点
    ntsRandom rand(random_seed + tid);

    int input_idx = blockIdx.x;
    if(input_idx >= input_node_count) {
        return;
    }

    // 该线程负责的顶点的起始位置
    VertexId_CUDA nid = destinations[input_idx];    // 获取输入节点的id
    int64_t start = global_column_offset[nid];
    int64_t end = global_column_offset[nid + 1];

    int neighbor_count = (int)(end - start);
    // std::printf("tid: %d, max sample: %d neighbor count: %d\n", tid, max_sample_count, neighbor_count);
    if(neighbor_count <= 0) {
        return;
    }

    // 如果邻居少于采样数量，全邻居采样
    int offset = sample_offset[input_idx];
    if(neighbor_count <= max_sample_count) {
        for(int sample_id = threadIdx.x; sample_id < neighbor_count; sample_id += blockDim.x) {
            VertexId_CUDA gid = global_row_indices[start + sample_id];
            sample_indices[offset + sample_id] = gid;
            src_index[gid] = layer + 1;
        }
        return;
    }

    uint64_t sample_per_thread[ITEMS_PER_THREAD];
    int M = max_sample_count;
    int N = neighbor_count;

    typedef cub::BlockRadixSort<uint64_t, BLOCK_DIM, ITEMS_PER_THREAD> BlockRadixSort;
    struct SampleSharedData{
        int s[BLOCK_DIM * ITEMS_PER_THREAD];
        int p[BLOCK_DIM * ITEMS_PER_THREAD];
        int q[BLOCK_DIM * ITEMS_PER_THREAD];
        int chain[BLOCK_DIM * ITEMS_PER_THREAD];
        int last_chain_tmp[BLOCK_DIM * ITEMS_PER_THREAD];
    };


    __shared__ union {
        typename BlockRadixSort::TempStorage temp_storage;
        SampleSharedData sample_shared_data;
    } shared_data;


    // 生成随机数
#pragma unroll
    for(int i = 0; i < ITEMS_PER_THREAD; i++) { // 每个线程负责一个顶点的部分邻居
        uint32_t idx = i * BLOCK_DIM + threadIdx.x;
        uint32_t r = idx < M ? rand.rand(N - idx) : N;    // 高32位存储生成的随机数
        if(r != N)
//         std::printf("tid: %d, random num: %d,  max: %d\n", tid, r, N-idx);
        sample_per_thread[i] = ((uint64_t)r << 32ul) | idx; // 低32位存储负责的线程的id
    }
    __syncthreads();

//     std::printf("tid: %d debug 1\n", tid);

    // 对生成的随机数进行排序
    // s, p = parallel_sort(r), 其中r就是sample_per_thread, s就是生成的随机数，p就是对应的线程的id
    // 如果每个线程负责一个随机数的话，那么p就是一个线性数组
    BlockRadixSort(shared_data.temp_storage).SortBlockedToStriped(sample_per_thread);
    __syncthreads();

#pragma unroll
    for(int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = i * BLOCK_DIM + threadIdx.x;
        int s = (sample_per_thread[i] >> 32ul); // 获得生成的随机数,高32位
        shared_data.sample_shared_data.s[idx] = s;
        int p = sample_per_thread[i] & 0xFFFFFFFF; // 获取该数对应的线程id，低32位
        shared_data.sample_shared_data.p[idx] = p;
        if(idx < M) {
            shared_data.sample_shared_data.q[p] = idx;
        }
        shared_data.sample_shared_data.chain[idx] = idx;    // chain被赋值位一个线性数组

    }
    __syncthreads();

#pragma unroll
    for(int i = 0; i <ITEMS_PER_THREAD; i++) {
        int idx = i * BLOCK_DIM + threadIdx.x;
        int si = shared_data.sample_shared_data.s[idx];
        int si1 = shared_data.sample_shared_data.s[idx+1];
        if(idx < M && (idx == M - 1 || si != si1) && si >= N-M) {
            shared_data.sample_shared_data.chain[N-si-1] = shared_data.sample_shared_data.p[idx];
        }
    }
    __syncthreads();

    for(int step = 0; step < Log2UpCUDA(M); step++){    // Log2UpCUDA计算log2的向上取整值
#pragma unroll
        for(int i  = 0; i < ITEMS_PER_THREAD; i++) {
            int idx = i * BLOCK_DIM + threadIdx.x;
            shared_data.sample_shared_data.last_chain_tmp[idx] = shared_data.sample_shared_data.chain[idx];
        }
        __syncthreads();

#pragma unroll
        for(int i = 0; i < ITEMS_PER_THREAD; i++) {
            int idx = i * BLOCK_DIM + threadIdx.x;
            if(idx < M) {
                shared_data.sample_shared_data.chain[idx] =
                        shared_data.sample_shared_data.last_chain_tmp[shared_data.sample_shared_data.last_chain_tmp[idx]];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for(int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = i * BLOCK_DIM + threadIdx.x;
        shared_data.sample_shared_data.last_chain_tmp[idx] = N - shared_data.sample_shared_data.chain[idx] - 1;
    }
    __syncthreads();

#pragma unroll
    for(int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = i * BLOCK_DIM + threadIdx.x;
        int ai;
        if(idx < M) {
            int qi = shared_data.sample_shared_data.q[idx];
            if(idx == 0 || qi == 0
               || shared_data.sample_shared_data.s[qi] != shared_data.sample_shared_data.s[qi-1]) {
                ai = shared_data.sample_shared_data.s[qi];
            } else {
                int prev_i = shared_data.sample_shared_data.p[qi - 1];
                ai = shared_data.sample_shared_data.last_chain_tmp[prev_i];
            }
            sample_per_thread[i] = ai;
        }
    }
    __syncthreads();

#pragma unroll
    for(int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = i * BLOCK_DIM + threadIdx.x;
        int ai = sample_per_thread[i];
        if(idx < M) {
            assert(ai < N);
            VertexId_CUDA gid = global_row_indices[start + ai];
            src_index[gid] = layer + 1;
            sample_indices[offset + idx] = gid;
        }
    }
}



__global__ void sample_processing_traverse_gpu_kernel_stage3(
									VertexId_CUDA *src_index,
									VertexId_CUDA src_index_size,
									VertexId_CUDA* src,
									VertexId_CUDA* src_count,
									VertexId_CUDA layer){

	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
	const int WARPSIZE = 32;
	size_t laneId = threadId % WARPSIZE;
	size_t warp_id = threadId / WARPSIZE;
	for(long i = threadId; i < src_index_size; i += blockDim.x*gridDim.x){
       	if(src_index[i] == layer + 1){
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
