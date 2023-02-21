/*
Copyright (c) 2021-2022 Qiange Wang, Northeastern University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "ntsCUDA.hpp"
#include <vector>
#include <map>
#include <algorithm>
#ifndef COOCSC_HPP
#define COOCSC_HPP

class sampCSC{
public:  
    sampCSC(){
        v_size=0;
        e_size=0;
        size_dev_src=0;
        size_dev_dst=0;
        size_dev_src_max = 0;
        size_dev_dst_max = 0;
        size_dev_edge = 0;
        size_dev_edge_max = 0;
        column_offset.clear();    
        row_indices.clear();
        src_index.clear();
        destination.clear();
        source.clear();
    }
    sampCSC(VertexId v_, VertexId e_){
        v_size=v_;
        e_size=e_;
        size_dev_src=0;
        size_dev_dst=0;
        size_dev_src_max = 0;
        size_dev_dst_max = 0;
        size_dev_edge = 0;
        size_dev_edge_max = 0;
        column_offset.clear();    
        row_indices.clear();
        src_index.clear();
        destination.clear();
        source.clear();
    }
    sampCSC(VertexId v_){
        v_size=v_;
        e_size=0;
        size_dev_src=0;
        size_dev_dst=0;
        size_dev_src_max = 0;
        size_dev_dst_max = 0;
        size_dev_edge = 0;
        size_dev_edge_max = 0;
        column_offset.clear();    
        row_indices.clear();
        src_index.clear();
        destination.clear();
        source.clear();
    }
    ~sampCSC(){
        v_size=0;
        e_size=0;
        column_offset.clear();    
        row_indices.clear();
        src_index.clear();
        destination.clear();
        source.clear();
    }
  

    void csc_to_csr(){
        // row_offset.clear();
        // row_offset.resize(src_size + 1,0);
        row_offset = std::vector<VertexId>(src_size + 1 , 0);

        #pragma omp parallel for
        for(VertexId i = 0; i < e_size; i++){
            row_offset[row_indices[i]]++;
        }
        for(VertexId i = 0, comsum = 0; i < src_size; i++){
            VertexId temp = row_offset[i];
            row_offset[i] = comsum;
            comsum += temp;
        }
        row_offset[src_size] = e_size;
        //std::vector<int> tmp_row_offset(row_offset.begin(), row_offset.end());
        for(VertexId i = 0; i < v_size; i++){
            for(VertexId j = column_offset[i]; j < column_offset[i + 1]; j++){
                VertexId col = row_indices[j];
                VertexId dest = row_offset[col];
                column_indices[dest] = i;
                row_offset[col]++;
            }
        }
        for(VertexId col = 0, last = 0; col <= src_size; col++){
            VertexId temp = row_offset[col];
            row_offset[col] = last;
            last = temp;
        }
    }

    void allocate_vertex(){
        destination.resize(v_size,0);       
        column_offset.resize(v_size+1,0);
    }
    void allocate_dst(VertexId v_size_){
        v_size=v_size_;
        destination.resize(v_size,0);
    }
    void allocate_vertex(VertexId v_size_){
        v_size=v_size_;
        destination.resize(v_size,0);
        column_offset.resize(v_size+1,0);
    }
    void allocate_co_from_dst(){
        v_size=destination.size();
        column_offset.resize(v_size+1,0);
    }
    void allocate_edge(){
        assert(0);
        row_indices.resize(e_size,0);
        column_indices.resize(e_size,0);
        edge_weight_backward.resize(e_size,0.0);
        edge_weight_forward.resize(e_size,0.0);
        sample_ans.resize(e_size,0);
    }
    void allocate_edge(VertexId e_size_){
        e_size=e_size_;
        row_indices.resize(e_size,0);
        column_indices.resize(e_size,0);
        edge_weight_backward.resize(e_size,0.0);
        edge_weight_forward.resize(e_size,0.0);
        sample_ans.resize(e_size,0);
    }
    void allocate_all(){
        allocate_vertex();
        allocate_edge();
    }
    // void allocate_dev_array(VertexId vtx_size, VertexId edge_size){
    //     column_offset = (VertexId *)cudaMallocPinned((vtx_size + 1) * sizeof(VertexId));
    //     //dev_column_offset = (VertexId *)cudaMallocGPU((vtx_size + 1) * sizeof(VertexId));
    //     row_offset = (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId));
    //     //dev_row_offset = (VertexId *)cudaMallocGPU((edge_size + 1) * sizeof(VertexId));

    //     row_indices = (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId));
    //     edge_weight_forward = (ValueType *)cudaMallocPinned((edge_size + 1) * sizeof(ValueType));
    //     //dev_row_indices = (VertexId *)cudaMallocGPU((edge_size + 1) * sizeof(VertexId));
    //     //dev_edge_weight_forward = (ValueType *)cudaMallocGPU((edge_size + 1) * sizeof(ValueType));

    //     column_indices = (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId)); 
    //     edge_weight_backward = (ValueType *)cudaMallocPinned((edge_size + 1) * sizeof(ValueType)); 
    //     //dev_column_indices = (VertexId *)cudaMallocGPU((edge_size + 1) * sizeof(VertexId));
    //     //dev_edge_weight_backward = (ValueType *)cudaMallocGPU((edge_size + 1) * sizeof(ValueType));
    // }

    void allocate_dev_array(){
        if(column_offset.size()>size_dev_dst_max){
            if(size_dev_dst_max!=0){
                FreeEdge(dev_column_offset);
                FreeEdge(dev_destination);
            }      
            size_dev_dst_max = column_offset.size() * 1.2;
            allocate_gpu_edge(&dev_column_offset, size_dev_dst_max);
            allocate_gpu_edge(&dev_destination, size_dev_dst_max);
            size_dev_dst = column_offset.size() - 1;
        }
        else{
            size_dev_dst = column_offset.size() - 1;
        }

        if(row_indices.size()>size_dev_edge_max){
            if(size_dev_edge_max != 0){
                FreeEdge(dev_row_indices);
                FreeEdge(dev_column_indices);
                FreeBuffer(dev_edge_weight_backward);
                FreeBuffer(dev_edge_weight_forward);
            }      
            size_dev_edge_max = row_indices.size() * 1.2;
            allocate_gpu_edge(&dev_row_indices, size_dev_edge_max);
            allocate_gpu_edge(&dev_column_indices, size_dev_edge_max);
            allocate_gpu_buffer(&dev_edge_weight_backward, size_dev_edge_max);
            allocate_gpu_buffer(&dev_edge_weight_forward, size_dev_edge_max);
            size_dev_edge = row_indices.size();
        }
        else{
            size_dev_edge = row_indices.size();
        }

        if(row_offset.size()>size_dev_src_max){
            if(size_dev_src_max!=0){
                FreeEdge(dev_row_offset);
                FreeEdge(dev_source);
            }      
            size_dev_src_max = row_offset.size() * 1.2;
            allocate_gpu_edge(&dev_row_offset, size_dev_src_max);
            allocate_gpu_edge(&dev_source, size_dev_src_max);
            size_dev_src = row_offset.size() - 1;
        }
        else{
            size_dev_src = row_offset.size() - 1;
        }

    }
    void allocate_dev_array_async(cudaStream_t stream){
        if(column_offset.size()>size_dev_dst_max){
            if(size_dev_dst_max!=0){
                FreeEdgeAsync(dev_column_offset, stream);
                FreeEdgeAsync(dev_destination, stream);
            }      
            size_dev_dst_max = column_offset.size() * 1.2;
            allocate_gpu_edge_async(&dev_column_offset, size_dev_dst_max, stream);
            allocate_gpu_edge_async(&dev_destination, size_dev_dst_max, stream);
            size_dev_dst = column_offset.size() - 1;
        }
        else{
            size_dev_dst = column_offset.size() - 1;
        }

        if(row_indices.size()>size_dev_edge_max){
            if(size_dev_edge_max != 0){
                FreeEdgeAsync(dev_row_indices, stream);
                FreeEdgeAsync(dev_column_indices, stream);
                FreeBufferAsync(dev_edge_weight_backward, stream);
                FreeBufferAsync(dev_edge_weight_forward, stream);
            }      
            size_dev_edge_max = row_indices.size() * 1.2;
            allocate_gpu_edge_async(&dev_row_indices, size_dev_edge_max, stream);
            allocate_gpu_edge_async(&dev_column_indices, size_dev_edge_max, stream);
            allocate_gpu_buffer_async(&dev_edge_weight_backward, size_dev_edge_max, stream);
            allocate_gpu_buffer_async(&dev_edge_weight_forward, size_dev_edge_max, stream);
            size_dev_edge = row_indices.size();
        }
        else{
            size_dev_edge = row_indices.size();
        }

        if(row_offset.size()>size_dev_src_max){
            if(size_dev_src_max!=0){
                FreeEdge(dev_row_offset);
                FreeEdge(dev_source);
            }      
            size_dev_src_max = row_offset.size() * 1.2;
            allocate_gpu_edge(&dev_row_offset, size_dev_src_max);
            allocate_gpu_edge(&dev_source, size_dev_src_max);
            size_dev_src = row_offset.size() - 1;
        }
        else{
            size_dev_src = row_offset.size() - 1;
        }

    }
    void copy_data_to_device(){
        // printf("copy_size %d %d %d %d %d %d \n",size_dev_co,size_dev_ri,size_dev_ewf,size_dev_ci,size_dev_ro,size_dev_ewb);
        // printf("host size:%d %d %d %d %d %d \n",column_offset.size(),row_indices.size(),edge_weight_forward.size(),column_indices.size(),row_offset.size(),edge_weight_backward.size());
        move_bytes_in(dev_column_offset, &(column_offset[0]), (size_dev_dst + 1) * sizeof(VertexId));
        move_bytes_in(dev_row_indices, &(row_indices[0]), size_dev_edge * sizeof(VertexId));
        move_bytes_in(dev_edge_weight_forward, &(edge_weight_forward[0]), size_dev_edge * sizeof(ValueType));
        move_bytes_in(dev_column_indices, &(column_indices[0]), size_dev_edge * sizeof(VertexId));
        move_bytes_in(dev_row_offset, &(row_offset[0]), (size_dev_src + 1) * sizeof(VertexId));
        move_bytes_in(dev_edge_weight_backward, &(edge_weight_backward[0]) , size_dev_edge * sizeof(ValueType));
        move_bytes_in(dev_source, &(source[0]), size_dev_src * sizeof(VertexId));
        move_bytes_in(dev_destination, &(destination[0]) , size_dev_dst * sizeof(VertexId));
        // dev_column_offset = (VertexId *)getDevicePointer(column_offset);
        // dev_row_indices = (VertexId *)getDevicePointer(row_indices);
        // dev_edge_weight_forward = (ValueType *)getDevicePointer(edge_weight_forward);

        // dev_row_offset = (VertexId *)getDevicePointer(row_offset);         ///
        // dev_column_indices = (VertexId *)getDevicePointer(column_indices); ///
        // dev_edge_weight_backward = (ValueType *)getDevicePointer(edge_weight_backward); ///
    }
    void copy_data_to_device_async(cudaStream_t stream){
        // printf("copy_size %d %d %d %d %d %d \n",size_dev_co,size_dev_ri,size_dev_ewf,size_dev_ci,size_dev_ro,size_dev_ewb);
        // printf("host size:%d %d %d %d %d %d \n",column_offset.size(),row_indices.size(),edge_weight_forward.size(),column_indices.size(),row_offset.size(),edge_weight_backward.size());
        move_bytes_in_async(dev_column_offset, &(column_offset[0]), (size_dev_dst + 1) * sizeof(VertexId), stream);
        move_bytes_in_async(dev_row_indices, &(row_indices[0]), size_dev_edge * sizeof(VertexId), stream);
        move_bytes_in_async(dev_edge_weight_forward, &(edge_weight_forward[0]), size_dev_edge * sizeof(ValueType), stream);
        move_bytes_in_async(dev_column_indices, &(column_indices[0]), size_dev_edge * sizeof(VertexId), stream);
        move_bytes_in_async(dev_row_offset, &(row_offset[0]), (size_dev_src + 1) * sizeof(VertexId), stream);
        move_bytes_in_async(dev_edge_weight_backward, &(edge_weight_backward[0]) , size_dev_edge * sizeof(ValueType), stream);
        move_bytes_in_async(dev_source, &(source[0]), size_dev_src * sizeof(VertexId), stream);
        move_bytes_in_async(dev_destination, &(destination[0]) , size_dev_dst * sizeof(VertexId), stream);
        // dev_column_offset = (VertexId *)getDevicePointer(column_offset);
        // dev_row_indices = (VertexId *)getDevicePointer(row_indices);
        // dev_edge_weight_forward = (ValueType *)getDevicePointer(edge_weight_forward);

        // dev_row_offset = (VertexId *)getDevicePointer(row_offset);         ///
        // dev_column_indices = (VertexId *)getDevicePointer(column_indices); ///
        // dev_edge_weight_backward = (ValueType *)getDevicePointer(edge_weight_backward); ///
    }
    void WeightCompute(std::function<ValueType(VertexId, VertexId)> weight_compute){
      #pragma omp parallel for
      for(VertexId i = 0; i < src_size; i++){
            for(VertexId j = row_offset[i]; j < row_offset[i + 1]; j++){
                VertexId v_dst = column_indices[j];
                VertexId v_src = i;
                VertexId v_src_m = source[v_src];
                VertexId v_dst_m = destination[v_dst];
                edge_weight_backward[j] =
                    weight_compute(v_src_m, v_dst_m);
            }
        }
      #pragma omp parallel for
        for(VertexId i = 0; i < v_size; i++){
            for(VertexId j = column_offset[i]; j < column_offset[i + 1]; j++){
                VertexId v_dst = i;
                VertexId v_src = row_indices[j];
                VertexId v_src_m = source[v_src];
                VertexId v_dst_m = destination[v_dst];
                edge_weight_forward[j] =
                    weight_compute(v_src_m, v_dst_m);
            }
        }
    }
    
    VertexId c_o(VertexId vid){
        return column_offset[vid];
    }
    VertexId r_i(VertexId vid){
        return row_indices[vid];
    }
    std::vector<VertexId>& dst(){
        return destination;
    }
    std::vector<VertexId>& src(){
        return source;
    }
    std::vector<VertexId>& c_o(){
        return column_offset;
    }
    std::vector<VertexId>& r_i(){
        return row_indices;
    }
    std::vector<ValueType>& e_w_f(){
        return edge_weight_forward;
    }
    std::vector<VertexId>& c_i(){
        return column_indices;
    }
    std::vector<VertexId>& r_o(){
        return row_offset;
    }
    std::vector<ValueType>& e_w_b(){
        return edge_weight_backward;
    }
    VertexId* dev_dst(){
        return dev_destination;
    }
    VertexId* dev_src(){
        return dev_source;
    }
    VertexId* dev_c_o(){
        return dev_column_offset;
    }
    VertexId* dev_r_i(){
        return dev_row_indices;
    }
    ValueType* dev_e_w_f(){
        return dev_edge_weight_forward;
    }
    VertexId* dev_c_i(){
        return dev_column_indices;
    }
    VertexId* dev_r_o(){
        return dev_row_offset;
    }
    ValueType* dev_e_w_b(){
        return dev_edge_weight_backward;
    }
    ValueType* dev_e_w(){
        return edge_weight;
    }
    VertexId get_distinct_src_size(){
        return src_size;
    }
    VertexId get_distinct_dst_size(){
        return v_size;
    }
    // void debug(){
    //     printf("print one layer:\ndst:\t");
    //     for(int i=0;i<destination.size();i++){
    //         printf("%d\t",destination[i]);
    //     }printf("\nc_o:\t");
    //     for(int i=0;i<column_offset.size();i++){
    //         printf("%d\t",column_offset[i]);
    //     }printf("\nr_i:\t");
    //     for(int i=0;i<row_indices.size();i++){
    //         printf("%d\t",row_indices[i]);
    //     }printf("\nrid:\t");
    //     for(int i=0;i<row_indices_debug.size();i++){
    //         printf("%d\t",row_indices_debug[i]);
    //     }printf("\nsrc:\t");
    //     for(int i=0;i<source.size();i++){
    //         printf("%d\t",source[i]);
    //     }printf("\n\n");
    // }
    
std::vector<VertexId> row_indices_debug;//local id

std::vector<VertexId> sample_ans;//local id
std::vector<VertexId> column_offset;//local offset    
std::vector<VertexId> row_indices;//local id
std::vector<VertexId> row_offset;
std::vector<VertexId> column_indices;//local id
std::vector<VertexId> source;//global id
std::vector<VertexId> destination;//global_id
std::vector<ValueType> edge_weight_forward;//local id
std::vector<ValueType> edge_weight_backward;//local id

VertexId* dev_destination;
VertexId* dev_source;
VertexId  size_dev_src,size_dev_dst;
VertexId  size_dev_src_max,size_dev_dst_max;

VertexId* dev_column_offset;
VertexId* dev_row_indices;
ValueType* dev_edge_weight_forward;
VertexId* dev_column_indices;
VertexId* dev_row_offset;
ValueType* dev_edge_weight_backward;
VertexId  size_dev_edge;
VertexId  size_dev_edge_max;
ValueType* edge_weight;

std::map<VertexId,VertexId> src_index;//set

VertexId v_size; //dst_size
VertexId e_size; // edge size
VertexId src_size;//distinct src size
};



#endif