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
#ifndef FULLLYREPGRAPH_HPP
#define FULLLYREPGRAPH_HPP
#include <assert.h>
#include <map>
#include <math.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include "core/graph.hpp"
#include "core/coocsc.hpp"


class SampledSubgraph{
public:    

    CSC_segment_pinned* graph_chunk;  
    
    SampledSubgraph(){
        //threads = std::max(numa_num_configured_cpus() - 1, 1);
        threads = 32;
    }
    SampledSubgraph(int layers_, int batch_size_,std::vector<int>& feature_size,std::vector<int>& fanout_,
                    Graph<Empty> *graph){
        layers=layers_;
        batch_size=batch_size_;
        fanout=fanout_;
        sampled_sgs.clear();
        curr_layer=0;
        for(int i=0;i<layers;i++){
            sampled_sgs.push_back(new sampCSC(0));
        }
        threads = std::max(numa_num_configured_cpus() - 1, 1) ;
    }
    
    SampledSubgraph(int layers_,std::vector<int>& fanout_){
        layers=layers_;
        fanout=fanout_;
        sampled_sgs.clear();
        curr_layer=0;
        for(int i=0;i<layers;i++){
            sampled_sgs.push_back(new sampCSC(0));
        }
        threads = std::max(numa_num_configured_cpus() - 1, 1);
        //threads = 1;
    }

    SampledSubgraph(int layers_,std::vector<int>& fanout_,bool gpu_){
        layers=layers_;
        fanout=fanout_;
        sampled_sgs.clear();
        curr_layer=0;
        gpu = gpu_;
        threads = std::max(numa_num_configured_cpus() - 1, 1);
        //threads = 1;
        //printf("threads:%d\n",threads);
    }
  static void print_cuda_use()
    {
        size_t free_byte;
        size_t total_byte;
    
        cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    
        if (cudaSuccess != cuda_status) {
            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
            exit(1);
        }
    
        double free_db = (double)free_byte;
        double total_db = (double)total_byte;
        double used_db_1 = (total_db - free_db) / 1024.0 / 1024.0;
        std::cout << "Now used GPU memory " << used_db_1 << "  MB\n";
    }
    SampledSubgraph(int layers_,std::vector<int>& fanout_,VertexId all_vertices_,Cuda_Stream* cs_){
        layers=layers_;
        fanout=fanout_;
        all_vertices = all_vertices_;
        cs = cs_;
        sampled_sgs.clear();
        curr_layer=0;
        src_index.resize(layers,0);
        for(int i = 0; i < layers; i++){
            sampled_sgs.push_back(new sampCSC(0));
        allocate_gpu_edge(&(src_index[i]), all_vertices);
        }
        global_buffer_used=0;
//        global_buffer_capacity=1024*1024*128*2;
        global_buffer_capacity=1024*1024*10*2;
        global_value_used = 0;
//        global_value_capacity = 1024*1024*128;
        global_value_capacity = 1024*1024*10;

        // std::printf("线程: %d 分配了1：%ld, stream: %p\n", omp_get_thread_num(), sizeof(float)*(global_value_capacity), cs->stream);
        allocate_gpu_buffer_async(&global_value_buffer, global_value_capacity, cs->stream);//for weight
        // std::printf("线程: %d 分配了2：%ld\n", omp_get_thread_num(), sizeof(VertexId_CUDA)*(global_buffer_capacity));
        allocate_gpu_edge_async(&global_data_buffer, global_buffer_capacity, cs->stream);
        allocate_gpu_edge_async(&queue_count, 1, cs->stream);


        // std::printf("线程: %d 分配了3：%ld\n", omp_get_thread_num(), sizeof(VertexId_CUDA)*(all_vertices));
        allocate_gpu_edge_async(&outdegree, all_vertices, cs->stream);
        // std::printf("线程: %d 分配了4：%ld\n", omp_get_thread_num(), sizeof(VertexId_CUDA)*(all_vertices));
        allocate_gpu_edge_async(&indegree, all_vertices, cs->stream);
        threads = std::max(numa_num_configured_cpus() - 1, 1);

    }

//     template<class DevType> void free_gpu_memory(DevType* dev_ptr) {
// #if CUDA_ENABLE
//     CHECK_CUDA_RESULT(cudaFree(dev_ptr));
// #else
//         printf("CUDA DISABLED free_gpu_memory\n");
//         exit(0);
// #endif
//     }

    ~SampledSubgraph(){
        fanout.clear();
        for(int i=0;i<sampled_sgs.size();i++){
            delete sampled_sgs[i];
        }
        sampled_sgs.clear();

        cudaFree(global_data_buffer);
        cudaFree(global_data_buffer);
        cudaFree(queue_count);
        cudaFree(outdegree);
        cudaFree(indegree);

    }
    void update_degrees(Graph<Empty> *graph, int layer) {

        VertexId* outs = graph->out_degree_for_backward;
        VertexId* ins = graph->in_degree_for_backward;
#pragma omp parallel for
        for (int i = 0; i < graph->vertices; ++i) {
            outs[i] = 0;
            ins[i] = 0;
        }
        VertexId v_size = sampled_sgs[layer]->dst().size();
        for(VertexId i = 0; i < v_size; i++){
            ins[sampled_sgs[layer]->dst()[i]] += sampled_sgs[layer]->c_o(i+1) - sampled_sgs[layer]->c_o(i);
            for(VertexId j = sampled_sgs[layer]->c_o(i); j < sampled_sgs[layer]->c_o(i+1); j++){
                VertexId v_src = sampled_sgs[layer]->r_i(j);
                VertexId v_src_m = sampled_sgs[layer]->src()[v_src];
                outs[v_src_m]++;
            }
        }
    }

    void move_degree_to_gpu(VertexId* cpu_in_degree, VertexId* cpu_out_degree, VertexId vertexs){
        cs->move_degree_to_gpu(cpu_in_degree, cpu_out_degree, indegree, outdegree, vertexs);
    }
    
    void update_degrees_GPU(int layer) {
        VertexId v_size = sampled_sgs[layer]->v_size;
        cs->ReFreshDegree(outdegree,indegree,all_vertices);
        cs->UpdateDegree(outdegree,indegree,v_size,sampled_sgs[layer]->dev_dst(),
                         sampled_sgs[layer]->dev_src(),
                         sampled_sgs[layer]->dev_c_o(),sampled_sgs[layer]->dev_r_i());
    }

    void Get_Weight(int layer){
        VertexId v_size = sampled_sgs[layer]->v_size;
        cs->GetWeight(sampled_sgs[curr_layer]->edge_weight,outdegree,indegree,v_size,
                      sampled_sgs[layer]->dev_dst(),sampled_sgs[layer]->dev_src(),
                      sampled_sgs[layer]->dev_c_o(),sampled_sgs[layer]->dev_r_i());
    }
    void reset(int layers_,std::vector<int>& fanout_){
        layers = layers_;
        fanout = fanout_;
        curr_layer = 0;
        for(int i = 0;i<sampled_sgs.size();i++){
            delete sampled_sgs[i];
        }
        sampled_sgs.clear();
        //sampled_sgs.resize(layers,new sampCSC(0));
    }
    
    /**
     * @brief  初始化第一层采样
     * @note   
     * @param  destination: 需要采样的节点数组开始的位置
     * @param  batch_size_: 采样的batch大小，节点的数量
     * @retval None
     */
    void gpu_init_first_layer(VertexId* destination, int batch_size_){
        curr_layer = 0;
        curr_dst_size = batch_size_;
        global_buffer_used = 0;
        global_value_used = 0;
        //LOG_INFO("global_buffer_used %ld|global_buffer_capacity %ld",global_buffer_used,global_buffer_capacity);
        assert(global_buffer_used < global_buffer_capacity);
        sampled_sgs[curr_layer]->v_size = curr_dst_size;
        sampled_sgs[curr_layer]->dev_destination = global_data_buffer + global_buffer_used;

        VertexId* tmp_ptr = global_data_buffer + curr_dst_size;
//        std::printf("curr dis size: %d, need: %d, global data buffer before: %p, after: %p\n", curr_dst_size, sizeof(VertexId) * curr_dst_size, global_data_buffer, tmp_ptr);
        global_buffer_used += curr_dst_size;
        sampled_sgs[curr_layer]->dev_column_offset = global_data_buffer + global_buffer_used;
        //sampled_sgs[curr_layer]->size_dev_co = curr_dst_size + 1;
        global_buffer_used += curr_dst_size+1;
        // 将顶点数据移动到设备端
        // move_bytes_in(sampled_sgs[curr_layer]->dev_destination, destination, (curr_dst_size) * sizeof(VertexId));
        move_bytes_in_async(sampled_sgs[curr_layer]->dev_destination, destination, (curr_dst_size) * sizeof(VertexId), cs->stream);
    }

    /**
     * @brief  初始化其它层的参数
     * @note   
     * @param  layer: 该层是第几层
     * @retval None
     *
    **/
    void gpu_init_proceeding_layer(int layer){
        curr_layer = layer;
        curr_dst_size = sampled_sgs[curr_layer-1]->src_size;
        assert(global_buffer_used < global_buffer_capacity);
        sampled_sgs[curr_layer]->v_size = curr_dst_size;
        sampled_sgs[curr_layer]->dev_destination = sampled_sgs[curr_layer-1]->dev_source;
        //sampled_sgs[curr_layer]->dev_destination=global_data_buffer+global_buffer_used;
        //global_buffer_used+=curr_dst_size;
        sampled_sgs[curr_layer]->dev_column_offset = global_data_buffer + global_buffer_used;
        //sampled_sgs[curr_layer]->size_dev_co = curr_dst_size + 1;
        global_buffer_used += curr_dst_size + 1;
    }
    
    /**
     * @description: 初始化column_offset
     * @param {int} layer 当前层
     * @param {VertexId} src_index_size 传进来的是所有节点
     * @param {VertexId*} global_column_offset 全局的column_offset
     * @param {VertexId*} tmp_data_buffer 用于暂存数据的GPU缓存
     * @param {Cuda_Stream*} cs 运行该操作的GPU流
     * @return {*}
     */
    void gpu_sampling_init_co(
                    int layer,
                    VertexId src_index_size,
                    VertexId* global_column_offset,
                    VertexId* tmp_data_buffer, Cuda_Stream* cs,
                    VertexId& edge_size){
        cs->sample_processing_get_co_gpu(sampled_sgs[curr_layer]->dev_destination,
                                         sampled_sgs[curr_layer]->dev_column_offset,
                                         global_column_offset,
                                         sampled_sgs[curr_layer]->v_size,
                                         tmp_data_buffer,
                                         src_index_size,
                                         queue_count,
                                         src_index[layer],
                                         fanout[layer],
                                         edge_size);
    }

    void gpu_sampling_init_co_omit(
                    int layer,
                    VertexId* CacheFlag,
                    VertexId src_index_size,
                    VertexId* global_column_offset,
                    VertexId* tmp_data_buffer, Cuda_Stream* cs,
                    VertexId& edge_size){
        cs->sample_processing_get_co_gpu_omit(
                                         CacheFlag,
                                         sampled_sgs[curr_layer]->dev_destination,
                                         sampled_sgs[curr_layer]->dev_column_offset,
                                         global_column_offset,
                                         sampled_sgs[curr_layer]->v_size,
                                         tmp_data_buffer,
                                         src_index_size,
                                         queue_count,
                                         src_index[layer],
                                         fanout[layer],
                                         edge_size);
    }

    void gpu_sampling(int layer,
                VertexId* global_column_offset,
                VertexId* global_row_indices,
                VertexId  whole_vertex_size,double &test_time,
                Cuda_Stream* cs){
        VertexId* edge_size=new VertexId[2];
       
        //move_bytes_out(edge_size, global_data_buffer + global_buffer_used - 1, sizeof(VertexId)); //edge_size
        //sampled_sgs[layer]->e_size = edge_size[0];
//        sampled_sgs[layer]->e_size = fanout[layer] * sampled_sgs[layer]->v_size;//
        edge_size[1] = 0;
        
        // if(sampled_sgs[layer]->e_size > sampled_sgs[layer]->size_dev_edge_max){
        //     if(sampled_sgs[layer]->size_dev_edge_max != 0){
        //         FreeBuffer(sampled_sgs[layer]->edge_weight);
        //     }      
        //     sampled_sgs[layer]->size_dev_edge_max = sampled_sgs[layer]->e_size * 1.2;
        //     allocate_gpu_buffer(&sampled_sgs[layer]->edge_weight, sampled_sgs[layer]->size_dev_edge_max);
        // }
        
        sampled_sgs[layer]->edge_weight = global_value_buffer + global_value_used;
        global_buffer_used += sampled_sgs[layer]->e_size;
        
        sampled_sgs[layer]->dev_row_indices = global_data_buffer + global_buffer_used;
        global_buffer_used += sampled_sgs[layer]->e_size;
        sampled_sgs[layer]->dev_source = global_data_buffer + global_buffer_used;
        test_time -= get_time();
        // move_bytes_in(queue_count, edge_size + 1, sizeof(VertexId)); //clear queue_count
        move_bytes_in_async(queue_count, edge_size + 1, sizeof(VertexId), cs->stream);
        
        
        cs->sample_processing_traverse_gpu(
                sampled_sgs[layer]->dev_destination,
                sampled_sgs[layer]->dev_column_offset,
                sampled_sgs[layer]->dev_row_indices,
                global_column_offset,
                global_row_indices,
                src_index[layer],
                sampled_sgs[layer]->v_size,
                sampled_sgs[layer]->e_size,
                whole_vertex_size,
                sampled_sgs[layer]->dev_source,
                queue_count,
                layer,
                fanout[layer]);
        
        //get src_size;
        
        // move_bytes_out(edge_size + 1, queue_count, sizeof(VertexId)); 
        move_bytes_out_async(edge_size + 1, queue_count, sizeof(VertexId), cs->stream); 
        test_time += get_time();
         
        sampled_sgs[layer]->src_size=edge_size[1];
        
        cs->sample_processing_update_ri_gpu(
                                sampled_sgs[layer]->dev_row_indices,
                                src_index[layer],
                                sampled_sgs[layer]->e_size,
                                whole_vertex_size);  
        global_buffer_used += sampled_sgs[layer]->src_size;   
        
       // LOG_INFO("edge_size %d, src_size %d",edge_size[0],edge_size[1]);                   
        //5) using cub to compact the array.and get the unique number,
        //6) then update the row_indice_array
        //7) update sampled_sgs[layer]->src_size
        //8) update src
        
        delete [] edge_size;
       
    }

    void sample_load_destination(std::function<void(std::vector<VertexId> &destination)> dst_select,VertexId layer){
        dst_select(sampled_sgs[layer]->dst());//init destination;
    }

    void init_co_only(std::function<VertexId(VertexId dst)> get_nbr_size,VertexId layer){
        curr_dst_size= sampled_sgs[layer]->dst().size();
        VertexId offset=0;
        for(VertexId i=0;i<curr_dst_size;i++){
            sampled_sgs[layer]->c_o()[i]=offset;
            offset+=get_nbr_size(sampled_sgs[layer]->dst()[i]);//init destination;
        }
        sampled_sgs[layer]->c_o()[curr_dst_size]=offset;  
        sampled_sgs[layer]->allocate_edge(offset);
    }

    
//     void sample_load_destination(VertexId layer){
//         assert(layer>0);
//         for(VertexId i_id=0;i_id<curr_dst_size;i_id++){
//             sampled_sgs[layer]->dst()[i_id]=sampled_sgs[layer-1]->src()[i_id];
//         }
//     }
    
//     void sample_load_destination1(VertexId layer){
//         assert(layer>0);
//  //       sampled_sgs[layer]->destination=sampled_sgs[layer-1]->src();
//         VertexId v_size=sampled_sgs[layer-1]->src().size();
//         sampled_sgs[layer]->allocate_dst(v_size);
//         memcpy(&(sampled_sgs[layer]->dst()[0]),
//                     &(sampled_sgs[layer-1]->src()[0]),
//                         sizeof(VertexId)*v_size);
//     }
    void sample_processing(std::function<void(VertexId fanout_i,
                VertexId dst,
                    std::vector<VertexId>& column_offset,
                        std::vector<VertexId>& row_indices,VertexId id)> vertex_sample){
        {
            omp_set_num_threads(threads);
#pragma omp parallel for num_threads(threads)
            for (VertexId begin_v_i = 0;begin_v_i < curr_dst_size;begin_v_i += 1) {
            // for every vertex, apply the sparse_slot at the partition
            // corresponding to the step
             vertex_sample(fanout[curr_layer],
                    sampled_sgs[curr_layer]->dst()[begin_v_i],
                     sampled_sgs[curr_layer]->c_o(),
                      sampled_sgs[curr_layer]->r_i(),
                        begin_v_i);
            }
        }
        
    }

    void sample_processing1(std::function<void(VertexId fanout_i,
                VertexId dst,
                    std::vector<VertexId>& column_offset,
                        std::vector<VertexId>& row_indices,VertexId id)> vertex_sample, int layer_){
        {
          //  omp_set_dynamic(0);
          //printf("sample_processing1 threads %d\n",cpu_threads_);
          omp_set_num_threads(threads);
#pragma omp parallel for
            for (VertexId begin_v_i = 0;begin_v_i < curr_dst_size;begin_v_i += 1) {
                // if(omp_get_thread_num() == 0) {
                //     std::printf("线程数量: %d\n", omp_get_num_threads());
                // }
            // for every vertex, apply the sparse_slot at the partition
            // corresponding to the step
             vertex_sample(fanout[layer_],
                    sampled_sgs[layer_]->dst()[begin_v_i],
                     sampled_sgs[layer_]->c_o(),
                      sampled_sgs[layer_]->r_i(),
                        begin_v_i);
            }
        }
        
    } 

    // void sample_postprocessing(int layer){
    //     sampled_sgs[layer]->postprocessing();
    //     if(gpu)
    //         sampled_sgs[layer]->csc_to_csr();
    //     curr_dst_size=sampled_sgs[layer]->get_distinct_src_size();
    //     curr_layer++;
    // }
    
    void compute_one_layer(std::function<void(VertexId local_dst, 
                          std::vector<VertexId>& column_offset, 
                              std::vector<VertexId>& row_indices)>sparse_slot,VertexId layer){
        
        {
//            std::printf("threads: %d\n", threads);
            omp_set_num_threads(threads);
#pragma omp parallel for num_threads(threads)
            for (VertexId begin_v_i = 0;
                begin_v_i < sampled_sgs[layer]->dst().size();
                    begin_v_i += 1) {
                    sparse_slot(begin_v_i,sampled_sgs[layer]->c_o(),sampled_sgs[layer]->r_i());
            }
        }
    }

    void compute_one_layer(std::function<void(VertexId local_dst, 
                          std::vector<VertexId>& column_offset, 
                              std::vector<VertexId>& row_indices)>sparse_slot,VertexId layer,std::vector<VertexId> cacheflag){
        
        {
            omp_set_num_threads(threads);
#pragma omp parallel for
            for (VertexId begin_v_i = 0;
                begin_v_i < sampled_sgs[layer]->dst().size();
                    begin_v_i += 1) {
                    if(!cacheflag[sampled_sgs[layer]->dst()[begin_v_i]])
                    sparse_slot(begin_v_i,sampled_sgs[layer]->c_o(),sampled_sgs[layer]->r_i());
            }
        }
    }

    void compute_one_layer_batch(std::function<void(VertexId local_dst, std::vector<VertexId>& column_offset,
                                                    std::vector<VertexId>& row_indices)> sparse_slot, VertexId layer,
                                 VertexId batch_start, VertexId batch_end) {
#pragma omp parallel for num_threads(threads)
        for(VertexId begin_v_i = batch_start; begin_v_i < batch_end; begin_v_i+=1){
            sparse_slot(begin_v_i, sampled_sgs[layer]->c_o(), sampled_sgs[layer]->r_i());
        }
    }

    std::vector<sampCSC*> sampled_sgs;
    int layers;
    int batch_size;
    std::vector<int> fanout;
    int curr_layer;
    int curr_dst_size;
    int threads;
    bool gpu;

    VertexId* global_data_buffer;
    ValueType* global_value_buffer;
    long global_value_used;
    long global_value_capacity;
    long global_buffer_used;
    long global_buffer_capacity;
    VertexId all_vertices;
    std::vector<VertexId*> src_index; 
    VertexId *queue_count;

    Cuda_Stream* cs;
    VertexId* outdegree;
    VertexId* indegree;

};
class FullyRepGraph{
public:
    //topo:
  VertexId *dstList;
  VertexId *srcList;
  //meta info
  Graph<Empty> *graph_;
  VertexId *partition_offset;
  VertexId partitions;
  VertexId partition_id;
  VertexId global_vertices;
  VertexId global_edges;
  // vertex range for this chunk
  VertexId owned_vertices;
  VertexId owned_edges;
  VertexId owned_mirrors;
  
  //global graph;
  VertexId* column_offset;
  VertexId* row_indices;
  
  FullyRepGraph(){
  }
  FullyRepGraph(Graph<Empty> *graph){
        global_vertices=graph->vertices;
        global_edges=graph->edges;
        owned_vertices=graph->owned_vertices;
        partitions=graph->partitions;
        partition_id=graph->partition_id;
        partition_offset=graph->partition_offset;
        graph_=graph;
  }
  void SyncAndLog(const char* data){
      MPI_Barrier(MPI_COMM_WORLD);
      if(partition_id==0)
      std::cout<<data<<std::endl;
  }
  void GenerateAll(){
     
        ReadRepGraphFromRawFile();
        SyncAndLog("NeutronStar::Preprocessing[Generate Full Replicated Graph Topo]");
     SyncAndLog("------------------finish graph preprocessing--------------\n");
  }
   void ReadRepGraphFromRawFile() {
    column_offset=new VertexId[global_vertices+1];
    row_indices=new VertexId[global_edges];   
    memset(column_offset, 0, sizeof(VertexId) * (global_vertices + 1));
    memset(row_indices, 0, sizeof(VertexId) * global_edges);
    VertexId *tmp_offset = new VertexId[global_vertices + 1];
    memset(tmp_offset, 0, sizeof(VertexId) * (global_vertices + 1));
    long total_bytes = file_size(graph_->filename.c_str());
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0) {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
#endif
    int edge_unit_size = sizeof(VertexId)*2;
    EdgeId read_edges = global_edges;
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = 0;
    long read_bytes;
    int fin = open(graph_->filename.c_str(), O_RDONLY);
    EdgeUnit<Empty> *read_edge_buffer = new EdgeUnit<Empty>[CHUNKSIZE];

    assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes =
            read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes =
            read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      // #pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        tmp_offset[dst + 1]++;
      }
    }
    for (int i = 0; i < global_vertices; i++) {
      tmp_offset[i + 1] += tmp_offset[i];
    }

    memcpy(column_offset, tmp_offset, sizeof(VertexId) * (global_vertices + 1));
    // printf("%d\n", column_offset[vertices]);
    assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes =
            read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes =
            read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      // #pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        //        if(dst==875710)
        //            printf("%d",read_edge_buffer[e_i].src);
        row_indices[tmp_offset[dst]++] = src;
      }
    }
    delete []read_edge_buffer;
    delete []tmp_offset; 
  } 
};



#endif