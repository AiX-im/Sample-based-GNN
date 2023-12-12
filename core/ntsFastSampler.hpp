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
#ifndef NTSFASTSAMPLER_HPP
#define NTSFASTSAMPLER_HPP
#include <mutex>
#include <cmath>
#include <random>
#include <stdlib.h>
#include "FullyRepGraph.hpp"
#include "cuda/ntsCUDA.hpp"
#include "NtsScheduler.hpp"


enum class WeightType{Sum, Mean, None};
class FastSampler{
public:
        double pre_pro_time;
        double pro_time;
        double post_pro_time;
        double copy_gpu_time;
        double all_time;
        double test_time;
        double init_co_time = 0;
        double init_layer_time = 0;
    FullyRepGraph* whole_graph;
    Graph<Empty> *graph;
    VertexId work_range[2];
    VertexId work_offset;
    int layer;
    std::vector<int> fanout;
    std::vector<VertexId> sample_nids;
    int gpu_id;
    bool gpu_sampler;
    SampledSubgraph* ssg;// excepted to be single write multi read
    SampledSubgraph** ssgs;
      
    //cpu-based data;
    Bitmap* samp_bitmap;
    std::vector<VertexId> src_index_array;  // 该变量相当于GPU的source_index, 大小为总顶点数量
    // VertexId* cacheflag;
    // VertexId* cacheepoch;

    // VertexId* dev_cacheflag;
    // VertexId* dev_cacheepoch;
    int epoch_gap = 2;
    bool to_gpu;
    int cpu_threads;

    //GPU-based data;
    VertexId* column_offset;//on device memory
    VertexId* tmp_data_buffer;//on device memory
    VertexId* batch_out_degree;
    VertexId* batch_in_degree;
    VertexId* row_indices;// on pinned memory
    ValueType* edge_weight;// on pinned memory
    VertexId* dev_row_indices;// view
    ValueType* dev_edge_weight;// view
    Cuda_Stream * cs;

    FastSampler(Graph<Empty> *graph_,FullyRepGraph* whole_graph_, std::vector<VertexId>& index,//cpu
            int layers_,std::vector<int> fanout_,int batch_size,bool to_gpu_=false,int gpu_id_=0, 
            int pipeline_num = 1, Cuda_Stream* cudaStreamArray = nullptr){
        assert(index.size() > 0);
        sample_nids.assign(index.begin(), index.end());
        assert(sample_nids.size() == index.size());
        whole_graph=whole_graph_;
        graph = graph_;
        work_range[0]=0;
        work_range[1]=sample_nids.size();
        work_offset=0;
        layer =  layers_;
        fanout = fanout_;
        samp_bitmap=new Bitmap(whole_graph->global_vertices);
        src_index_array.resize(whole_graph->global_vertices, 0);
        
        if(pipeline_num <= 1) {
            pipeline_num = 1;
        }
        ssgs = new SampledSubgraph*[pipeline_num];
        // TODO: 这里可以改，现在这个选择不一定好
        if(cudaStreamArray == nullptr) {
            for(int i = 0; i < pipeline_num; i++) {
                ssgs[i] =new SampledSubgraph(layer,batch_size,graph->gnnctx->layer_size,fanout,graph); //cpu sampler
            }
        } else {
            for(int i = 0; i < pipeline_num; i++) {
                ssgs[i] = new SampledSubgraph(layer,fanout,whole_graph->global_vertices, &cudaStreamArray[i]);
            }
        }
        ssg = ssgs[0];
        to_gpu=to_gpu_;
        cpu_threads = std::max(numa_num_configured_cpus() - 1, 1);
        gpu_id=gpu_id_;
        gpu_sampler=false;
        pre_pro_time=0.0;
        pro_time=0.0;
        post_pro_time=0.0;
        all_time=0.0;
        copy_gpu_time=0.0;
        test_time = 0.0;
        cs=new Cuda_Stream();

        // int vtx_size, edge_size;
        // edge_size = batch_size;
        // for(int i = 0; i < layer; i++){
        //     vtx_size = edge_size;
        //     edge_size = vtx_size * fanout[i];
        //     ssg->sampled_sgs[i]->allocate_dev_array(vtx_size,edge_size);
        // }
    }

    FastSampler(FullyRepGraph* whole_graph_, std::vector<VertexId>& index,//GPU
            int layers_, int batch_size_, std::vector<int> fanout_, int pipeline_num = 1, Cuda_Stream* cuda_stream = nullptr){
        assert(index.size() > 0);
        sample_nids.assign(index.begin(), index.end());
        assert(sample_nids.size() == index.size());
        whole_graph=whole_graph_;
        graph = whole_graph->graph_;
        work_range[0]=0;
        work_range[1]=sample_nids.size();
        work_offset=0;
        layer =  layers_;
        fanout = fanout_;
        gpu_sampler=true;
        if(cuda_stream == nullptr) {
            cs = new Cuda_Stream();
            cuda_stream = cs;
        } else {
            cs = cuda_stream;
        }
        if(pipeline_num <= 1) {
            pipeline_num = 1;
        }
        ssgs = new SampledSubgraph*[pipeline_num];
        for(int i = 0; i < pipeline_num; i++){
            ssgs[i] = new SampledSubgraph(layer,batch_size_,fanout,whole_graph->global_vertices,&cuda_stream[i]);//gpu sampler
            ssgs[i]->move_degree_to_gpu(graph->in_degree_for_backward, graph->out_degree_for_backward, graph->vertices);
        }
        ssg = ssgs[0];
        
    //    ssg = new SampledSubgraph(layer,fanout,whole_graph->global_vertices,cs);//gpu sampler
        //whole graph to gpu
        allocate_gpu_edge(&column_offset, whole_graph->global_vertices+1);
        allocate_gpu_edge(&tmp_data_buffer, whole_graph->global_vertices+1);
//        allocate_gpu_edge(&dev_row_indices, whole_graph->global_edges);
        row_indices=whole_graph->row_indices;
        //edge_weight=(ValueType* )cudaMallocPinned(((long)whole_graph->global_edges)*sizeof(ValueType));
        move_bytes_in(column_offset, whole_graph->column_offset, (whole_graph->global_vertices+1)*sizeof(VertexId));
        //move_bytes_in(dev_row_indices, whole_graph->row_indices, ((whole_graph->global_edges)*sizeof(VertexId)));
//        memcpy(row_indices, whole_graph->row_indices,(whole_graph->global_edges)*sizeof(VertexId));
        //memcpy(edge_weight, whole_graph->edge_weight,(whole_graph->global_edges)*sizeof(ValueType));
        
        dev_row_indices=(VertexId *)getDevicePointer(row_indices);
        // dev_row_indices = (VertexId *)cudaMallocGPU(sizeof(VertexId) * ((long)whole_graph->global_edges));
        // move_bytes_in(dev_row_indices, row_indices, sizeof(VertexId) * ((long)whole_graph->global_edges), false);
        //dev_edge_weight=(ValueType *)getDevicePointer(edge_weight);
            pre_pro_time=0.0;
            pro_time=0.0;
            post_pro_time=0.0;
            all_time=0.0;
            copy_gpu_time=0.0;
            test_time = 0.0;
    }

    /**
     * GPU采样将dst添加到src中
     * @param sg_num 子图的数量，实际上是流水线的数量
     */
    void set_merge_src_dst(int sg_num){
        for(int i = 0; i < sg_num; i++) {
            for(int j = 0; j < ssgs[i]->sampled_sgs.size(); j++) {
                ssgs[i]->sampled_sgs[j]->set_merge_src_dst();
            }
        }
    }

    /**
     * GPU采样将dst添加到src中
     * @param sg_num 子图的数量，实际上是流水线的数量
     */
    void set_cpu_merge_src_dst(){
        for(int i = 0; i < ssg->sampled_sgs.size(); i++) {
            ssg->sampled_sgs[i]->set_merge_src_dst();
        }
    }

    int random_uniform_int(const int min = 0, const int max = 1) {
        // thread_local std::default_random_engine generator;
        static thread_local std::mt19937 generator(2000);
        std::uniform_int_distribution<int> distribution(min, max);
        return distribution(generator);
    }

//    void init_cache(){
//        cacheflag = (VertexId*)cudaMallocPinned((long)(whole_graph->global_vertices)*sizeof(VertexId));
//        cacheepoch = (VertexId*)cudaMallocPinned((long)(whole_graph->global_vertices)*sizeof(VertexId));
//        dev_cacheflag = (VertexId*)getDevicePointer(cacheflag);
//        dev_cacheepoch = (VertexId*)getDevicePointer(cacheepoch);
//
//        omp_set_num_threads(cpu_threads);
//        #pragma omp parallel for
//            for(VertexId vid = 0; vid < whole_graph->global_vertices; vid++){
//                cacheepoch[vid] = 0;
//                int nbrs = whole_graph->column_offset[vid + 1] - whole_graph->column_offset[vid];
//                if(nbrs < fanout[layer - 1] * 10){
//                    cacheflag[vid] = 0;
//                }
//                else{
//                    cacheflag[vid] = -1;
//                }
//        }
//    }

    void load_feature_gpu(NtsVar& local_feature,ValueType* global_feature_buffer){
        if(local_feature.size(0) < ssg->sampled_sgs[layer-1]->src_size){
            local_feature.resize_({
                    ssg->sampled_sgs[layer-1]->src_size,
                                        local_feature.size(1)});
        }
            //do the load operation.
        ValueType *local_feature_buffer =
        whole_graph->graph_->Nts->getWritableBuffer(
                local_feature, torch::DeviceType::CUDA);
        cs->zero_copy_feature_move_gpu(local_feature_buffer,
                                    global_feature_buffer,
                                    ssg->sampled_sgs[layer-1]->dev_source,
                                    local_feature.size(1),
                                    ssg->sampled_sgs[layer-1]->src_size);
    }

    void load_feature_gpu(Cuda_Stream * cudaStream, SampledSubgraph* subgraph, NtsVar& local_feature, ValueType* global_feature_buffer) {
        // std::printf("thread: %d, size: %d\n", omp_get_thread_num(), subgraph->sampled_sgs[0]->src_size);
//        std::printf("feature dim: %d\n", local_feature.dim());
        if(local_feature.size(0) != subgraph->sampled_sgs[layer-1]->src_size){
            local_feature.resize_({
                                          subgraph->sampled_sgs[layer-1]->src_size,
                                          local_feature.size(1)});
        }
        //do the load operation.
        ValueType *local_feature_buffer =
                whole_graph->graph_->Nts->getWritableBuffer(
                        local_feature, torch::DeviceType::CUDA);
        cudaStream->zero_copy_feature_move_gpu(local_feature_buffer,
                                       global_feature_buffer,
                                       subgraph->sampled_sgs[layer-1]->dev_source,
                                       local_feature.size(1),
                                       subgraph->sampled_sgs[layer-1]->src_size);
    }

    void load_feature_gpu_cache(Cuda_Stream* cuda_stream, SampledSubgraph* ssg,
                                                   NtsVar& local_feature, ValueType* global_feature_buffer,
                                                   ValueType* dev_cache_feature, VertexId* local_idx,
                                                   VertexId* local_idx_cache,
                                                   VertexId* cache_node_hashmap, VertexId* dev_local_idx, 
                                                   VertexId* dev_local_idx_cache,
                                                   VertexId* dev_cache_node_hashmap, VertexId* outmost_vertex) {
    // std::vector<int> &cache_node_hashmap) {

        auto csc_layer = ssg->sampled_sgs[layer-1];
        if (local_feature.size(0) != csc_layer->src_size) {
            local_feature.resize_({csc_layer->src_size, local_feature.size(1)});
        }
        ValueType* local_feature_buffer =
            whole_graph->graph_->Nts->getWritableBuffer(local_feature, torch::DeviceType::CUDA);

        // std::vector<int> local_idx_cache, local_idx;
        int local_idx_cnt = 0;
        int local_idx_cache_cnt = 0;
        
        // std::vector<int> local_idx_cache, global_idx_cache, local_idx, global_idx;
        cuda_stream->move_bytes_out(outmost_vertex, csc_layer->dev_source, (csc_layer->src_size));
        //LOG_DEBUG("src_size %d vertices %d", csc_layer->src_size, whole_graph->graph_->vertices);
        for (int i = 0; i < csc_layer->src_size; ++i) {
            int node_id = outmost_vertex[i];
            // LOG_DEBUG("node_id %d ", node_id);
            // LOG_DEBUG("cache_node_hashmap[node_id] %d", cache_node_hashmap[node_id]);
            if (cache_node_hashmap[node_id] != -1) {
                local_idx_cache[local_idx_cache_cnt++] = i;
                // local_idx_cache.push_back(cache_node_hashmap[node_id]);
                // global_idx_cache.push_back(csc_layer->src[i]);
            } else {
                local_idx[local_idx_cnt++] = i;
                // global_idx.push_back(csc_layer->src[i]);
            }
        }
        
        // //区分cache点与非cache点
        // cuda_stream->init_cache_map(csc_layer->dev_source, dev_cache_node_hashmap, dev_local_idx_cache, dev_local_idx, dev_local_idx_cache);

        // LOG_DEBUG("start zero_copy_feature_move_gpU_cache");
        double trans_feature_cost = -get_time();
        //dev_local_idx_cache[0] 非cache点个数
        cuda_stream->zero_copy_feature_move_gpu_cache(local_feature_buffer, global_feature_buffer, csc_layer->dev_source,
                                                    local_feature.size(1), local_idx_cnt, dev_local_idx);
        trans_feature_cost += get_time();
        // LOG_DEBUG("gather_fature_from_gpu_cache");
        double gather_gpu_cache_cost = -get_time();
        //dev_local_idx_cache[1] cache点个数
        cuda_stream->gather_feature_from_gpu_cache(local_feature_buffer, dev_cache_feature, csc_layer->dev_source,
                                                local_feature.size(1), local_idx_cache_cnt, dev_local_idx_cache,
                                                dev_cache_node_hashmap);
        gather_gpu_cache_cost += get_time();
        return;
    }


    void load_dst_src_feature_gpu(Cuda_Stream * cudaStream, SampledSubgraph* subgraph, NtsVar& local_feature,
                                  ValueType* global_feature_buffer) {
//        std::printf("local size: %d, dst size: %d, src size: %d\n", subgraph->sampled_sgs[layer-1]->local_size,
//                    subgraph->sampled_sgs[layer-1]->v_size, subgraph->sampled_sgs[layer-1]->src_size);

        if(local_feature.size(0) != subgraph->sampled_sgs[layer-1]->local_size){
            local_feature.resize_({subgraph->sampled_sgs[layer-1]->local_size,
                                          local_feature.size(1)});
        }
        //do the load operation.
        ValueType *local_feature_buffer =
                whole_graph->graph_->Nts->getWritableBuffer(
                        local_feature, torch::DeviceType::CUDA);

        cudaStream->zero_copy_feature_move_gpu(local_feature_buffer,
                                               global_feature_buffer,
                                               subgraph->sampled_sgs[layer-1]->dev_local_to_global,
                                               local_feature.size(1),
                                               subgraph->sampled_sgs[layer-1]->local_size);


    }

    void load_embedding_gpu(NtsVar& local_feature,ValueType* global_feature_buffer){
        if(local_feature.size(0)<ssg->sampled_sgs[0]->src_size){
            local_feature.resize_({
                    ssg->sampled_sgs[0]->src_size,
                                        local_feature.size(1)});
        }
            //do the load operation.
        ValueType *local_feature_buffer =
        whole_graph->graph_->Nts->getWritableBuffer(
                local_feature, torch::DeviceType::CUDA);
        cs->zero_copy_feature_move_gpu(local_feature_buffer,
                                    global_feature_buffer,
                                    ssg->sampled_sgs[0]->dev_source,
                                    local_feature.size(1),
                                    ssg->sampled_sgs[0]->src_size);
    }

    //code by aix from 304 to 319
    NtsVar load_aggresult(NtsVar& aggresult){
        int feature_size = aggresult.size(1);
std::printf("load_aggresult 1 \n");
        NtsVar f_output = graph->Nts->NewKeyTensor({ssg->sampled_sgs[0]->v_size , feature_size}, torch::DeviceType::CPU);
        ValueType *f_input_buffer = graph->Nts->getWritableBuffer(aggresult, torch::DeviceType::CPU);
        ValueType *f_output_buffer = graph->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
        std::printf("load_aggresult 2 \n");
        int threads = std::max(1, numa_num_configured_cpus() / 2 - 1);
        omp_set_num_threads(threads);
        #pragma omp parallel for
        for (int i = 0; i < ssg->sampled_sgs[0]->v_size; i++) {
            memcpy(f_output_buffer + i * feature_size, f_input_buffer + ssg->sampled_sgs[0]->destination[i] * feature_size,
                feature_size * sizeof(ValueType));
        }
        std::printf("load_aggresult 3 \n");
        return f_output;
    }


    void load_embedding_gpu_local(NtsVar& local_feature,ValueType* global_feature_buffer){
            //do the load operation.
        ValueType *local_feature_buffer = whole_graph->graph_->Nts->getWritableBuffer(local_feature, torch::DeviceType::CUDA);
        cs->zero_copy_embedding_move_gpu(local_feature_buffer,
                                    global_feature_buffer,
                                    local_feature.size(1),
                                    ssg->sampled_sgs[0]->src_size);
    }


    void load_embedding_gpu_local(Cuda_Stream* cudaStream, SampledSubgraph* subgraph, NtsVar& local_feature,ValueType* global_feature_buffer){
        //do the load operation.
        ValueType *local_feature_buffer = whole_graph->graph_->Nts->getWritableBuffer(local_feature, torch::DeviceType::CUDA);
        cudaStream->zero_copy_embedding_move_gpu(local_feature_buffer,
                                         global_feature_buffer,
                                         local_feature.size(1),
                                         subgraph->sampled_sgs[0]->src_size);
    }


    void load_label_gpu(NtsVar& local_label, long *global_label_buffer){
            if(local_label.size(0)!=ssg->sampled_sgs[0]->v_size){
                local_label.resize_({ssg->sampled_sgs[0]->v_size});
            }
            long *local_label_buffer =
            whole_graph->graph_->Nts->getTensorBuffer1d<long>(
                local_label, torch::DeviceType::CUDA);
            //do the load operation.
            cs->global_copy_label_move_gpu(local_label_buffer,
                                           global_label_buffer,
                                           ssg->sampled_sgs[0]->dev_destination,
                                           ssg->sampled_sgs[0]->v_size);
    }

    void load_label_gpu(Cuda_Stream * cudaStream, SampledSubgraph* subgraph, NtsVar& local_label, long* global_label_buffer) {
        if(local_label.size(0)!=subgraph->sampled_sgs[0]->v_size){
            local_label.resize_({subgraph->sampled_sgs[0]->v_size});
        }
        long *local_label_buffer =
                whole_graph->graph_->Nts->getTensorBuffer1d<long>(
                        local_label, torch::DeviceType::CUDA);
        //do the load operation.
        cudaStream->global_copy_label_move_gpu(local_label_buffer,
                                       global_label_buffer,
                                       subgraph->sampled_sgs[0]->dev_destination,
                                       subgraph->sampled_sgs[0]->v_size);
    }

//    void updata_cacheflag(VertexId epoch){
//        omp_set_num_threads(cpu_threads);
//        int cachevtx = 0;
//        #pragma omp parallel for
//        for(VertexId vid = 0; vid < whole_graph->global_vertices; vid++){
//            if(cacheflag[vid] == 1 && cacheepoch[vid] < epoch - epoch_gap){
//                    cacheflag[vid] = 0;
//            }
//            // if(cacheflag[vid] == 1){
//            //     cachevtx++;
//            // }
//        }
//
//
//
//        // int cacheenable = 0;
//        // for(VertexId vid = 0; vid < ssg->sampled_sgs[1]->v_size; vid++){
//        //     VertexId id = ssg->sampled_sgs[1]->destination[vid];
//        //     if(cacheflag[id] == 1){
//        //         cacheenable++;
//        //     }
//        // }
//        // printf("v_size:%d,epoch:%d cachevtx:%d cacheenable:%d\n",ssg->sampled_sgs[1]->v_size, epoch,cachevtx,cacheenable);
//    }

    //cacheflag: 0  2
    void load_share_embedding(Cuda_Stream * cudaStream, 
                              SampledSubgraph* subgraph,
                              ValueType* dev_share_embedding,
                              NtsVar& dev_embedding,
                              VertexId *CacheMap,
                              VertexId *CacheFlag){
            ValueType *dev_embedding_buffer = whole_graph->graph_->Nts->getWritableBuffer(dev_embedding, torch::DeviceType::CUDA);
            cudaStream->dev_load_share_embedding(dev_embedding_buffer,
                                           dev_share_embedding,
                                           CacheFlag,
                                           CacheMap,
                                           dev_embedding.size(1),
                                           subgraph->sampled_sgs[layer-1]->dev_destination,
                                           subgraph->sampled_sgs[layer-1]->v_size);
    }


    //cacheflag: 0  2
    void load_share_embedding_and_feature(Cuda_Stream * cudaStream,
                              SampledSubgraph* subgraph,
                              ValueType* dev_share_feature,
                              ValueType* dev_share_embedding,
                              NtsVar& dev_feature,
                              NtsVar& dev_embedding,
                              VertexId *dev_cacheMap,
                              VertexId *dev_cacheFlag){
        ValueType *dev_feature_buffer = whole_graph->graph_->Nts->getWritableBuffer(dev_feature, torch::DeviceType::CUDA);
        ValueType *dev_embedding_buffer = whole_graph->graph_->Nts->getWritableBuffer(dev_embedding, torch::DeviceType::CUDA);
        cudaStream->dev_load_share_embedding_and_feature(dev_feature_buffer, dev_embedding_buffer,
                                             dev_share_feature, dev_share_embedding,
                                                         dev_cacheFlag,
                                                         dev_cacheMap,
                                             dev_feature.size(1), dev_embedding.size(1),
                                             subgraph->sampled_sgs[layer-1]->dev_destination,
                                             subgraph->sampled_sgs[layer-1]->v_size);
    }


    //cacheflag: 0  2
    void load_share_embedding_and_feature(Cuda_Stream * cudaStream,
                                          SampledSubgraph* subgraph,
                                          ValueType* dev_share_feature,
                                          ValueType* dev_share_embedding,
                                          NtsVar& dev_feature,
                                          NtsVar& dev_embedding,
                                          VertexId *dev_cacheMap,
                                          VertexId *dev_cacheLocation,
                                          VertexId super_batch_id){
        ValueType *dev_feature_buffer = whole_graph->graph_->Nts->getWritableBuffer(dev_feature, torch::DeviceType::CUDA);
        ValueType *dev_embedding_buffer = whole_graph->graph_->Nts->getWritableBuffer(dev_embedding, torch::DeviceType::CUDA);
        cudaStream->dev_load_share_embedding_and_feature(dev_feature_buffer, dev_embedding_buffer,
                                                         dev_share_feature, dev_share_embedding,
                                                         dev_cacheMap,
                                                         dev_cacheLocation,
                                                         dev_feature.size(1), dev_embedding.size(1),
                                                         subgraph->sampled_sgs[layer-1]->dev_destination,
                                                         subgraph->sampled_sgs[layer-1]->v_size, super_batch_id);
    }


    void load_share_embedding(Cuda_Stream * cudaStream,
                                          SampledSubgraph* subgraph,
                                          ValueType* dev_share_embedding,
                                          NtsVar& dev_embedding,
                                          VertexId *dev_cacheMap,
                                          VertexId *dev_cacheLocation,
                                          VertexId super_batch_id){
        ValueType *dev_embedding_buffer = whole_graph->graph_->Nts->getWritableBuffer(dev_embedding, torch::DeviceType::CUDA);
        cudaStream->dev_load_share_embedding(dev_embedding_buffer,
                                                         dev_share_embedding,
                                                         dev_cacheMap,
                                                         dev_cacheLocation,
                                                         dev_embedding.size(1),
                                                         subgraph->sampled_sgs[layer-1]->dev_destination,
                                                         subgraph->sampled_sgs[layer-1]->v_size, super_batch_id);
    }



    NtsVar get_X_mask(Cuda_Stream* cudaStream, SampledSubgraph* subgraph, VertexId* dev_cacheMap){
        auto vtx_num = subgraph->sampled_sgs[layer-1]->v_size;
//        NtsVar X_mask = torch::zeros({vtx_num, 1}, at::TensorOptions().dtype(torch::kBool).device_index(0));
//        auto X_mask_bool = X_mask.accessor<bool, 2>().data();
        std::printf("vtx_num: %d\n", vtx_num);
        uint8_t* X_mask_buffer;
        cudaMalloc(&X_mask_buffer, sizeof(uint8_t)*vtx_num);
        cudaStream->dev_get_X_mask(X_mask_buffer, subgraph->sampled_sgs[layer-1]->dev_destination, dev_cacheMap, vtx_num);
        auto X_mask = torch::from_blob(X_mask_buffer, {vtx_num, 1}, at::TensorOptions().dtype(torch::kBool).device_index(0));
        return X_mask;
    }

    void print_avg_weight(Cuda_Stream* cudaStream, SampledSubgraph* subgraph, VertexId* dev_cacheFlag) {
        float* cuda_sum;
        float cpu_sum = 0.0f;
        VertexId cache_num = 0;
        VertexId* dev_cache_num;
        cudaMalloc(&cuda_sum, sizeof(float));
        cudaMalloc(&dev_cache_num, sizeof(VertexId));
        cudaMemcpy(cuda_sum, &cpu_sum, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_cache_num, &cache_num, sizeof(VertexId), cudaMemcpyHostToDevice);
        auto* sg = subgraph->sampled_sgs[layer-1];
        auto* column_offset = sg->dev_c_o();
        auto* row_indices = sg->dev_r_i();
        cudaStream->dev_print_avg_weight(column_offset, row_indices, sg->dev_e_w(), sg->dev_destination, dev_cacheFlag, cuda_sum, dev_cache_num, sg->v_size);
        cudaMemcpy(&cpu_sum, cuda_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cache_num, dev_cache_num, sizeof(VertexId), cudaMemcpyDeviceToHost);
        std::printf("\tgpu cache weight sum: %f, edge num: %d, avg weight: %f\n", cpu_sum, cache_num, cpu_sum/cache_num);
    }

    //cacheflag: 0  2
    void load_share_aggregate(Cuda_Stream * cudaStream,
                              SampledSubgraph* subgraph,
                              ValueType* dev_share_feature,
                              NtsVar& dev_feature,
                              VertexId *CacheMap,
                              VertexId *CacheFlag){
        ValueType *dev_feature_buffer = whole_graph->graph_->Nts->getWritableBuffer(dev_feature, torch::DeviceType::CUDA);
        cudaStream->dev_load_share_aggregate(dev_feature_buffer,
                                                         dev_share_feature,
                                                         CacheFlag,
                                                         CacheMap,
                                                         dev_feature.size(1),
                                                         subgraph->sampled_sgs[layer-1]->dev_destination,
                                                         subgraph->sampled_sgs[layer-1]->v_size);
    }

    //cacheflag: 0  2
    void load_share_embedding(Cuda_Stream * cudaStream,
                              SampledSubgraph* subgraph,
                              ValueType* dev_share_embedding,
                              NtsVar& dev_embedding,
                              VertexId *CacheMap,
                              VertexId *CacheFlag,
                              NtsVar& X_mask_tensor,
                              uint8_t* dev_cache_mask){
        ValueType *dev_embedding_buffer = whole_graph->graph_->Nts->getWritableBuffer(dev_embedding, torch::DeviceType::CUDA);
        uint8_t* dev_x_mask = X_mask_tensor.packed_accessor<uint8_t, 2>().data();
        cudaStream->dev_load_share_embedding(dev_embedding_buffer,
                                             dev_share_embedding,
                                             CacheFlag,
                                             CacheMap,
                                             dev_embedding.size(1),
                                             subgraph->sampled_sgs[layer-1]->dev_destination,
                                             dev_x_mask,
                                             dev_cache_mask,
                                             subgraph->sampled_sgs[layer-1]->v_size);
    }


    void update_share_embedding(Cuda_Stream * cudaStream, 
                              SampledSubgraph* subgraph,
                              ValueType* dev_embedding,
                              ValueType* dev_share_embedding,   // 这个为NULL
                              VertexId *CacheMap,
                              VertexId *CacheFlag){

         //ValueType *dev_embedding_buffer = whole_graph->graph_->Nts->getWritableBuffer(dev_embedding, torch::DeviceType::CUDA);
//         std::printf("layer: %d, graph: %p\n", layer, graph);
//         std::printf("embedding size: %d\n", graph->gnnctx->layer_size[layer - 1]);
            cudaStream->dev_update_share_embedding(dev_embedding,
                                           dev_share_embedding,
                                           CacheMap,
                                           CacheFlag,
                                           graph->gnnctx->layer_size[layer - 1], //embedding size
                                           subgraph->sampled_sgs[layer-1]->dev_destination,
                                           subgraph->sampled_sgs[layer-1]->v_size);
    }


    void update_share_embedding_and_feature(Cuda_Stream * cudaStream,
                                SampledSubgraph* subgraph,
                                ValueType* dev_feature,
                                ValueType* dev_embedding,
                                ValueType* dev_share_aggregate,
                                ValueType* dev_share_embedding,
                                VertexId *CacheMap,VertexId *CacheFlag,
                                VertexId *dev_X_version, VertexId* dev_Y_version, VertexId require_version){

        //ValueType *dev_embedding_buffer = whole_graph->graph_->Nts->getWritableBuffer(dev_embedding, torch::DeviceType::CUDA);
//         std::printf("layer: %d, graph: %p\n", layer, graph);
//         std::printf("embedding size: %d\n", graph->gnnctx->layer_size[layer - 1]);

        cudaStream->dev_update_share_embedding_and_feature(dev_feature, dev_embedding, dev_share_aggregate,
                                               dev_share_embedding,
                                               CacheMap,
                                               CacheFlag,
                                               graph->gnnctx->layer_size[0],
                                               graph->gnnctx->layer_size[1], //embedding size
                                               subgraph->sampled_sgs[1]->dev_destination,
                                               dev_X_version, dev_Y_version,
                                               subgraph->sampled_sgs[1]->v_size, require_version);
    }


    SampledSubgraph* sample_gpu_fast(int batch_size_, WeightType weightType=WeightType::Sum){
        double tmp_all_time = 0.0;
        double tmp_pre_pro_time = 0.0;
        double tmp_pro_time = 0.0;
        tmp_all_time -= MPI_Wtime();
        assert(work_offset < work_range[1]);
        // 确定该batch的节点数量
        int actual_batch_size = std::min((VertexId)batch_size_,work_range[1]-work_offset);
        for(int i = 0; i < layer; i++){//debug
            tmp_pre_pro_time = 0.0;
            tmp_pre_pro_time -= MPI_Wtime();
            {
            init_layer_time -= MPI_Wtime();
            if(i == 0){
                ssg->gpu_init_first_layer(&(sample_nids[work_offset]),actual_batch_size);
            }else{
                ssg->gpu_init_proceeding_layer(i);
            }
            init_layer_time += MPI_Wtime();
            init_co_time -= MPI_Wtime();
            ssg->gpu_sampling_init_co(i,
                whole_graph->global_vertices,
                column_offset,
                tmp_data_buffer,
                ssg->cs,
                ssg->sampled_sgs[i]->e_size);
//            std::printf("start layer %d edge size: %d\n", i, ssg->sampled_sgs[i]->e_size);
            init_co_time += MPI_Wtime();
            }
            tmp_pre_pro_time+=MPI_Wtime();
            pre_pro_time+=tmp_pre_pro_time;
            

            tmp_pro_time=0.0;
            tmp_pro_time-=MPI_Wtime();
            ssg->gpu_sampling(i,
                    column_offset,
                    dev_row_indices,
                    whole_graph->global_vertices,test_time,
                    ssg->cs);
            tmp_pro_time+=MPI_Wtime();
            pro_time+=tmp_pro_time;
            post_pro_time -= MPI_Wtime();
            if(graph->config->up_degree){
                ssg->update_degrees_GPU(i);
//                ssg->update_degrees(graph,i);

            }
            if(weightType == WeightType::Mean){
                ssg->Get_Mean_Weight(i);
            } else if(weightType == WeightType::None){
              // WeightType为None不用生成weight，但是默认是sum的weightType，所以这样写
            } else {
                ssg->Get_Weight(i);
            }
            post_pro_time += MPI_Wtime();
        }
        tmp_all_time+=MPI_Wtime();
        all_time += tmp_all_time;
        work_offset += actual_batch_size;
        return ssg;
    }

    SampledSubgraph* sample_gpu_fast_omit(int batch_size_, VertexId* CacheFlag, WeightType weightType=WeightType::Sum){
        double tmp_all_time = 0.0;
        double tmp_pre_pro_time = 0.0;
        double tmp_pro_time = 0.0;
        tmp_all_time -= MPI_Wtime();
        assert(work_offset < work_range[1]);
        // 确定该batch的节点数量
        int actual_batch_size = std::min((VertexId)batch_size_,work_range[1]-work_offset);
        for(int i = 0; i < layer; i++){//debug
            tmp_pre_pro_time = 0.0;
            tmp_pre_pro_time -= MPI_Wtime();
            {
            init_layer_time -= MPI_Wtime();
            if(i == 0){
                ssg->gpu_init_first_layer(&(sample_nids[work_offset]),actual_batch_size);
//                std::printf("理论第0层采样节点:\n");
//                for(int j = 0; j < actual_batch_size; j++) {
//                    std::printf("%u ", sample_nids[work_offset + j]);
//                }
//                std::printf("\n");
            }else{
                ssg->gpu_init_proceeding_layer(i);
//                auto data_num = ssg->sampled_sgs[i]->v_size;
//                VertexId * data = new VertexId[data_num];
//                cudaMemcpy(data, ssg->sampled_sgs[i]->dev_destination, sizeof(VertexId) * data_num, cudaMemcpyDeviceToHost);
//                std::printf("理论第%d层采样节点:\n", i);
//                for(int j = 0; j < data_num; j++){
//                    std::printf("%u ", data[j]);
//                }
//                std::printf("\n");
//                delete[] data;

            }
            init_layer_time += MPI_Wtime();
            init_co_time -= MPI_Wtime();
            
            if(i < layer - 1){
                ssg->gpu_sampling_init_co(i,
                    whole_graph->global_vertices,
                    column_offset,
                    tmp_data_buffer,
                    ssg->cs,
                    ssg->sampled_sgs[i]->e_size);
            } 
            else{ //跳过 CacheFlag = 1 或 2 的bottom layer采样。
                ssg->gpu_sampling_init_co_omit(i,
                    CacheFlag,
                    whole_graph->global_vertices,
                    column_offset,
                    tmp_data_buffer,
                    ssg->cs,
                    ssg->sampled_sgs[i]->e_size);
            }           
            init_co_time += MPI_Wtime();
            }
            tmp_pre_pro_time+=MPI_Wtime();
            pre_pro_time+=tmp_pre_pro_time;
            

            tmp_pro_time=0.0;
            tmp_pro_time-=MPI_Wtime();
            ssg->gpu_sampling(i,
                    column_offset,
                    dev_row_indices,
                    whole_graph->global_vertices,test_time,
                    ssg->cs);
            tmp_pro_time+=MPI_Wtime();
            pro_time+=tmp_pro_time;
            post_pro_time -= MPI_Wtime();

            if(graph->config->up_degree){
//                ssg->update_degrees_GPU(i);
                ssg->update_cache_degrees_GPU(i);
//                ssg->update_degrees(graph,i);

            }
            if(weightType == WeightType::Mean){
                ssg->Get_Mean_Weight(i);
            } else if(weightType == WeightType::None){
                // WeightType为None不用生成weight，但是默认是sum的weightType，所以这样写
            } else {
                ssg->Get_Weight(i);
            }
            post_pro_time += MPI_Wtime();
        }
        tmp_all_time+=MPI_Wtime();
        all_time += tmp_all_time;
        work_offset += actual_batch_size;
        return ssg;
    }


    SampledSubgraph* sample_gpu_fast_omit(int batch_size_, VertexId* CacheFlag, VertexId super_batch_id, WeightType weightType=WeightType::Sum){
        double tmp_all_time = 0.0;
        double tmp_pre_pro_time = 0.0;
        double tmp_pro_time = 0.0;
        tmp_all_time -= MPI_Wtime();
        assert(work_offset < work_range[1]);
        // 确定该batch的节点数量
        int actual_batch_size = std::min((VertexId)batch_size_,work_range[1]-work_offset);
//        std::printf("sample layer: %d, batch size: %d, work_offset: %d, work_range[1]: %d\n",
//                    layer, actual_batch_size, work_offset, work_range[1]);

        for(int i = 0; i < layer; i++){//debug
            tmp_pre_pro_time = 0.0;
            tmp_pre_pro_time -= MPI_Wtime();
            {
                init_layer_time -= MPI_Wtime();
                if(i == 0){
//                    int deviceID;
//                    cudaGetDevice(&deviceID);
////                    printf("当前活动的GPU设备编号：%d\n", deviceID);
//                    std:printf("device %d thread 0x%lx worker offset: %d, work range[1]: %d, actual_batch_size: %d, real size: %d\n",
//                             deviceID, std::this_thread::get_id(), work_offset, work_range[1], actual_batch_size, sample_nids.size());
                    ssg->gpu_init_first_layer(&(sample_nids[work_offset]),actual_batch_size);
//                std::printf("理论第0层采样节点:\n");
//                for(int j = 0; j < actual_batch_size; j++) {
//                    std::printf("%u ", sample_nids[work_offset + j]);
//                }
//                std::printf("\n");
                }else{
                    ssg->gpu_init_proceeding_layer(i);
//                auto data_num = ssg->sampled_sgs[i]->v_size;
//                VertexId * data = new VertexId[data_num];
//                cudaMemcpy(data, ssg->sampled_sgs[i]->dev_destination, sizeof(VertexId) * data_num, cudaMemcpyDeviceToHost);
//                std::printf("理论第%d层采样节点:\n", i);
//                for(int j = 0; j < data_num; j++){
//                    std::printf("%u ", data[j]);
//                }
//                std::printf("\n");
//                delete[] data;

                }
//                // TODO: Toao用于debug
//                ssg->cs->CUDA_DEVICE_SYNCHRONIZE();

                init_layer_time += MPI_Wtime();
                init_co_time -= MPI_Wtime();

                if(i < layer - 1){
                    ssg->gpu_sampling_init_co(i,
                                              whole_graph->global_vertices,
                                              column_offset,
                                              tmp_data_buffer,
                                              ssg->cs,
                                              ssg->sampled_sgs[i]->e_size);
                }
                else{ //跳过 CacheFlag = super batch id 的bottom layer采样。
                    ssg->gpu_sampling_init_co_omit(i,
                                                   CacheFlag,
                                                   whole_graph->global_vertices,
                                                   column_offset,
                                                   tmp_data_buffer,
                                                   ssg->cs,
                                                   ssg->sampled_sgs[i]->e_size,
                                                   super_batch_id);
                }
//                // TODO: Toao用于debug
//                ssg->cs->CUDA_DEVICE_SYNCHRONIZE();

                init_co_time += MPI_Wtime();
            }
            tmp_pre_pro_time+=MPI_Wtime();
            pre_pro_time+=tmp_pre_pro_time;


            tmp_pro_time=0.0;
            tmp_pro_time-=MPI_Wtime();
            ssg->gpu_sampling(i,
                              column_offset,
                              dev_row_indices,
                              whole_graph->global_vertices,test_time,
                              ssg->cs);
            tmp_pro_time+=MPI_Wtime();
//            // TODO: Toao用于debug
//            LOG_INFO("after gpu sampling");
//            ssg->cs->CUDA_DEVICE_SYNCHRONIZE();
//            LOG_INFO("before update degree");

            pro_time+=tmp_pro_time;
            post_pro_time -= MPI_Wtime();

            if(graph->config->up_degree){
//                ssg->update_degrees_GPU(i);
                ssg->update_cache_degrees_GPU(i);
//                ssg->update_degrees(graph,i);

            }
            if(weightType == WeightType::Mean){
                ssg->Get_Mean_Weight(i);
            } else if(weightType == WeightType::None){
                // WeightType为None不用生成weight，但是默认是sum的weightType，所以这样写
            } else {
                ssg->Get_Weight(i);
            }
            // TODO: Toao用于debug
            ssg->cs->CUDA_DEVICE_SYNCHRONIZE();

            post_pro_time += MPI_Wtime();
        }
        tmp_all_time+=MPI_Wtime();
        all_time += tmp_all_time;
        work_offset += actual_batch_size;
        return ssg;
    }


    void set_ssg_num(int num, Cuda_Stream* cudaStreamArray){
        if(num <= 0) {
            return;
        }
        ssgs = new SampledSubgraph*[num];
        delete ssg;
        for(int i = 0; i < num; i++) {
            ssgs[i] = new SampledSubgraph(layer, fanout, whole_graph->global_vertices, &cudaStreamArray[i]);
            ssgs[i]->threads = std::max(ssgs[i]->threads - num, 1);
//            ssgs[i]->threads = ssgs[i]->threads;
        }

        ssg = ssgs[0];
    }
    SampledSubgraph* sample_fast(int batch_size_, int ssg_id) {
        ssg = ssgs[ssg_id];
        sample_fast(batch_size_);     
        return ssgs[ssg_id];
    }

    // SampledSubgraph* sample_gpu_fast(int batch_size_, int ssg_id) {
    //     ssg = ssgs[ssg_id];
    //     sample_gpu_fast(batch_size_);     
    //     return ssgs[ssg_id];
    // }

    SampledSubgraph* sample_gpu_fast(int batch_size_, int ssg_id,  WeightType weightType=WeightType::Sum) {
        ssg = ssgs[ssg_id];
        sample_gpu_fast(batch_size_);     
        return ssgs[ssg_id];
    }

    SampledSubgraph* sample_gpu_fast_omit(int batch_size_, int ssg_id, VertexId* CacheFlag, WeightType weightType=WeightType::Sum) {
        ssg = ssgs[ssg_id];
        sample_gpu_fast_omit(batch_size_, CacheFlag, weightType);
        return ssgs[ssg_id];
    }

    SampledSubgraph* sample_gpu_fast_omit(int batch_size_, int ssg_id, VertexId* CacheFlag, VertexId super_batch_id, WeightType weightType=WeightType::Sum) {
        ssg = ssgs[ssg_id];
        sample_gpu_fast_omit(batch_size_, CacheFlag, super_batch_id, weightType);
        return ssgs[ssg_id];
    }

    SampledSubgraph* sample_fast(int batch_size_, WeightType weightType = WeightType::Sum){
        double tmp_all_time=0.0;
        double tmp_pre_pro_time=0.0;
        double tmp_pro_time=0.0;
        double tmp_post_pro_time=0.0;
        double tmp_copy_gpu_time=0.0;
        // std::printf("线程: %d开始分配子图内存\n", omp_get_thread_num());
        // ssg = new SampledSubgraph(layer,fanout,whole_graph->global_vertices,cs);//gpu sampler
        // std::printf("\t线程: %d开始分配子图内存完毕\n", omp_get_thread_num());

        assert(work_offset<work_range[1]);
        int actual_batch_size=std::min((VertexId)batch_size_,work_range[1]-work_offset);
        //ssg->layer_size[0] = actual_batch_size;
        tmp_all_time-=MPI_Wtime();
        for(int i=0;i<layer;i++){
             //allocate vertex data
             //VertexId range_left=whole_graph->global_vertices;
             //VertexId range_right=0;
            tmp_pre_pro_time = 0.0;
            tmp_pre_pro_time-=MPI_Wtime();
            {
            if(i==0){
                ssg->sampled_sgs[i]->allocate_vertex(actual_batch_size);
                memcpy(&(ssg->sampled_sgs[i]->destination[0]),
                            &(sample_nids[work_offset]),
                                sizeof(VertexId)*actual_batch_size);
            }else{
                int actual_batch_size=ssg->sampled_sgs[i-1]->src().size();
                ssg->sampled_sgs[i]->allocate_vertex(actual_batch_size);
                memcpy(&(ssg->sampled_sgs[i]->destination[0]),
                            &(ssg->sampled_sgs[i-1]->src()[0]),
                                sizeof(VertexId)*actual_batch_size);
            }

            if(ssg->sampled_sgs[i]->is_merge_src_dst) {
                ssg->sampled_sgs[i]->dst_local_id.resize(ssg->sampled_sgs[i]->destination.size());
            }

            ssg->init_co_only([&](VertexId dst){
                VertexId nbrs = whole_graph->column_offset[dst + 1] - whole_graph->column_offset[dst];
                VertexId ret = std::min((int)nbrs, fanout[i]);
                if (ret == -1) {
                    ret = nbrs;
                }

                return ret;
            },i);
            }

            tmp_pre_pro_time+=MPI_Wtime();
            pre_pro_time+=tmp_pre_pro_time;

            tmp_pro_time = 0.0;
            tmp_pro_time -= MPI_Wtime();
            samp_bitmap->clear();
            //BFS
            {
            ssg->sample_processing1([&](VertexId fanout_i,
                    VertexId dst,
                    std::vector<VertexId> &column_offset,
                    std::vector<VertexId> &row_indices,VertexId id){
                        VertexId nbr_size = whole_graph->column_offset[dst+1] - whole_graph->column_offset[dst];
                        int num = column_offset[id + 1] - column_offset[id];
                        std::unordered_map<VertexId, int> sampled_idxs;
                        int pos = 0;
                        if(nbr_size > fanout_i){
                            while (sampled_idxs.size() < num) {        
                                VertexId rand = random_uniform_int(0,nbr_size - 1);
                                sampled_idxs.insert(std::pair<size_t, int>(rand, 1));
                            }
                            VertexId src_idx = whole_graph->column_offset[dst];
                            for (auto it = sampled_idxs.begin(); it != sampled_idxs.end(); it++) {
                                ssg->sampled_sgs[i]->sample_ans[column_offset[id] + pos] = whole_graph->row_indices[src_idx + it->first];
                                pos++;
                                samp_bitmap->set_bit(whole_graph->row_indices[src_idx + it->first]);
                            } 
                        }
                        else{
                            // 下面是非随机选的情况
                            VertexId src_idx = whole_graph->column_offset[dst];
                            for(VertexId idx = 0; idx < num; idx++){
                                ssg->sampled_sgs[i]->sample_ans[column_offset[id] + pos] = whole_graph->row_indices[src_idx + idx];
                                pos++;
                                samp_bitmap->set_bit(whole_graph->row_indices[src_idx + idx]);
                            }
                        }

                        if(ssg->sampled_sgs[i]->is_merge_src_dst){
                            samp_bitmap->set_bit(dst);
                        }
                                   
            }, i);
            }

            //whole_graph->SyncAndLog("finish processing");
            

            //postprocessing
            
            {
                //printf("#samp_bitmap size %d\n", samp_bitmap->size);
                int length=WORD_OFFSET(samp_bitmap->size)+1;
                ssg->sampled_sgs[i]->src_size = 0;
                //ssg->sampled_sgs[i]->src_index.clear();
                ssg->sampled_sgs[i]->source.clear();
                //ssg->sampled_sgs[i]->source = std::vector<VertexId>(src_size + 1 , 0)
                // 下面是将bitmap标记的src顶点放进source中
                for(VertexId i_src = 0; i_src < samp_bitmap->size; i_src += 64){
                    unsigned long word = samp_bitmap->data[WORD_OFFSET(i_src)];
                    VertexId vtx = i_src;
                    VertexId offset=0;
                    while(word != 0){
                        if(word & 1){
                            src_index_array[vtx+offset]=ssg->sampled_sgs[i]->src_size;
                            ssg->sampled_sgs[i]->source.push_back(vtx+offset);
                            ssg->sampled_sgs[i]->src_size++;
                        }
                        offset++;
                        word = word >> 1;   
                    }
                }
                //whole_graph->SyncAndLog("finish processing");
                ssg->sample_processing1([&](VertexId fanout_i,VertexId dst,
                    std::vector<VertexId> &column_offset,
                    std::vector<VertexId> &row_indices,VertexId id){
                        VertexId start = column_offset[id];
                        VertexId num = column_offset[id + 1] - column_offset[id];
                        for(VertexId pos = 0; pos < num; pos++){
                            VertexId src_idx = ssg->sampled_sgs[i]->sample_ans[start + pos];
                            row_indices[start + pos] = src_index_array[src_idx];
                        }
                        // TODO: 下面为dst也分配空间，dst_to_src(local)
                        if(ssg->sampled_sgs[i]->is_merge_src_dst) {
                            ssg->sampled_sgs[i]->dst_local_id[id] = src_index_array[ssg->sampled_sgs[i]->dst()[i]];
                        }

                }, i);
            }
            tmp_pro_time+=MPI_Wtime();
            pro_time+=tmp_pro_time;
            
            ssg->sampled_sgs[i]->csc_to_csr();
            tmp_post_pro_time = 0.0;
            tmp_post_pro_time -= MPI_Wtime();
            if(graph->config->up_degree){
                ssg->update_degrees(graph,i);

            }
            if(weightType == WeightType::Mean){
                ssg->sampled_sgs[i]->WeightCompute([&](VertexId src, VertexId dst) {
                    return nts::op::nts_norm_degree(graph,src,dst) / graph->in_degree_for_backward[dst];});
            } else if(weightType == WeightType::None){
                // WeightType为None不用生成weight，但是默认是sum的weightType，所以这样写
            } else {
                ssg->sampled_sgs[i]->WeightCompute([&](VertexId src, VertexId dst) {
                    return nts::op::nts_norm_degree(graph,src,dst);});
            }
            tmp_post_pro_time += MPI_Wtime();
            post_pro_time += tmp_post_pro_time; 

            //copy_to_device
            tmp_copy_gpu_time = 0.0;
            tmp_copy_gpu_time-=MPI_Wtime();
            if(to_gpu){                 
                ssg->sampled_sgs[i]->allocate_dev_array_async(ssg->cs->stream);
                ssg->sampled_sgs[i]->copy_data_to_device_async(ssg->cs->stream);
                // ssg->sampled_sgs[i]->allocate_dev_array();
                // ssg->sampled_sgs[i]->copy_data_to_device();
            }
            tmp_copy_gpu_time+=MPI_Wtime();
            copy_gpu_time+=tmp_copy_gpu_time;
        }

        tmp_all_time+=MPI_Wtime();
        all_time+=tmp_all_time;
        work_offset+=actual_batch_size;
        return ssg;
    }

    //code by aix from 1078 to 1084
    SampledSubgraph* sample_fast_allPD(int batch_size_, WeightType weightType = WeightType::Sum){
        assert(work_offset<work_range[1]);
        int actual_batch_size=std::min((VertexId)batch_size_,work_range[1]-work_offset);

        work_offset+=actual_batch_size;
        return ssg;
    }

    SampledSubgraph* sample_skip_fast(int batch_size_, int skip_batch_size, WeightType weightType = WeightType::Sum){
        double tmp_all_time=0.0;
        double tmp_pre_pro_time=0.0;
        double tmp_pro_time=0.0;
        double tmp_post_pro_time=0.0;
        double tmp_copy_gpu_time=0.0;
        // std::printf("线程: %d开始分配子图内存\n", omp_get_thread_num());
        // ssg = new SampledSubgraph(layer,fanout,whole_graph->global_vertices,cs);//gpu sampler
        // std::printf("\t线程: %d开始分配子图内存完毕\n", omp_get_thread_num());

        assert(work_offset<work_range[1]);
        int actual_batch_size=std::min((VertexId)batch_size_,work_range[1]-work_offset);
        //ssg->layer_size[0] = actual_batch_size;
        tmp_all_time-=MPI_Wtime();
        for(int i=0;i<layer;i++){
            //allocate vertex data
            //VertexId range_left=whole_graph->global_vertices;
            //VertexId range_right=0;
            tmp_pre_pro_time = 0.0;
            tmp_pre_pro_time-=MPI_Wtime();
            {
                if(i==0){
                    ssg->sampled_sgs[i]->allocate_vertex(actual_batch_size);
                    memcpy(&(ssg->sampled_sgs[i]->destination[0]),
                           &(sample_nids[work_offset]),
                           sizeof(VertexId)*actual_batch_size);
                }else{
                    int actual_batch_size=ssg->sampled_sgs[i-1]->src().size();
                    ssg->sampled_sgs[i]->allocate_vertex(actual_batch_size);
                    memcpy(&(ssg->sampled_sgs[i]->destination[0]),
                           &(ssg->sampled_sgs[i-1]->src()[0]),
                           sizeof(VertexId)*actual_batch_size);
                }
                ssg->init_co_only([&](VertexId dst){
                    VertexId nbrs = whole_graph->column_offset[dst + 1] - whole_graph->column_offset[dst];
                    VertexId ret = std::min((int)nbrs, fanout[i]);
                    if (ret == -1) {
                        ret = nbrs;
                    }

                    return ret;
                },i);
            }
            tmp_pre_pro_time+=MPI_Wtime();
            pre_pro_time+=tmp_pre_pro_time;

            tmp_pro_time = 0.0;
            tmp_pro_time -= MPI_Wtime();
            samp_bitmap->clear();
            //BFS
            {
                ssg->sample_processing1([&](VertexId fanout_i,
                                            VertexId dst,
                                            std::vector<VertexId> &column_offset,
                                            std::vector<VertexId> &row_indices,VertexId id){
                    VertexId nbr_size = whole_graph->column_offset[dst+1] - whole_graph->column_offset[dst];
                    int num = column_offset[id + 1] - column_offset[id];
                    std::unordered_map<VertexId, int> sampled_idxs;
                    int pos = 0;
                    if(nbr_size > fanout_i){
                        while (sampled_idxs.size() < num) {
                            VertexId rand = random_uniform_int(0,nbr_size - 1);
                            sampled_idxs.insert(std::pair<size_t, int>(rand, 1));
                        }
                        VertexId src_idx = whole_graph->column_offset[dst];
                        for (auto it = sampled_idxs.begin(); it != sampled_idxs.end(); it++) {
                            ssg->sampled_sgs[i]->sample_ans[column_offset[id] + pos] = whole_graph->row_indices[src_idx + it->first];
                            pos++;
                            samp_bitmap->set_bit(whole_graph->row_indices[src_idx + it->first]);
                        }
                    }
                    else{
                        VertexId src_idx = whole_graph->column_offset[dst];
                        for(VertexId idx = 0; idx < num; idx++){
                            ssg->sampled_sgs[i]->sample_ans[column_offset[id] + pos] = whole_graph->row_indices[src_idx + idx];
                            pos++;
                            samp_bitmap->set_bit(whole_graph->row_indices[src_idx + idx]);
                        }
                    }

                }, i);
            }

            //whole_graph->SyncAndLog("finish processing");


            //postprocessing

            {
                //printf("#samp_bitmap size %d\n", samp_bitmap->size);
                int length=WORD_OFFSET(samp_bitmap->size)+1;
                ssg->sampled_sgs[i]->src_size = 0;
                //ssg->sampled_sgs[i]->src_index.clear();
                ssg->sampled_sgs[i]->source.clear();
                //ssg->sampled_sgs[i]->source = std::vector<VertexId>(src_size + 1 , 0)
                for(VertexId i_src=0;i_src<samp_bitmap->size;i_src+=64){
                    unsigned long word= samp_bitmap->data[WORD_OFFSET(i_src)];
                    VertexId vtx=i_src;
                    VertexId offset=0;
                    while(word != 0){
                        if(word & 1){
                            src_index_array[vtx+offset]=ssg->sampled_sgs[i]->src_size;
                            ssg->sampled_sgs[i]->source.push_back(vtx+offset);
                            ssg->sampled_sgs[i]->src_size++;
                        }
                        offset++;
                        word = word >> 1;
                    }
                }
                //whole_graph->SyncAndLog("finish processing");
                ssg->sample_processing1([&](VertexId fanout_i,VertexId dst,
                                            std::vector<VertexId> &column_offset,
                                            std::vector<VertexId> &row_indices,VertexId id){
                    VertexId start = column_offset[id];
                    VertexId num = column_offset[id + 1] - column_offset[id];
                    for(VertexId pos = 0; pos < num; pos++){
                        VertexId src_idx = ssg->sampled_sgs[i]->sample_ans[start + pos];
                        row_indices[start + pos] = src_index_array[src_idx];
                    }
                }, i);
            }
            tmp_pro_time+=MPI_Wtime();
            pro_time+=tmp_pro_time;

            ssg->sampled_sgs[i]->csc_to_csr();
            tmp_post_pro_time = 0.0;
            tmp_post_pro_time -= MPI_Wtime();
            if(graph->config->up_degree){
                ssg->update_degrees(graph,i);

            }
            if(weightType == WeightType::Mean){
                ssg->sampled_sgs[i]->WeightCompute([&](VertexId src, VertexId dst) {
                    return nts::op::nts_norm_degree(graph,src,dst) / graph->in_degree_for_backward[dst];});
            } else if(weightType == WeightType::None){
                // WeightType为None不用生成weight，但是默认是sum的weightType，所以这样写
            } else {
                ssg->sampled_sgs[i]->WeightCompute([&](VertexId src, VertexId dst) {
                    return nts::op::nts_norm_degree(graph,src,dst);});
            }
            tmp_post_pro_time += MPI_Wtime();
            post_pro_time += tmp_post_pro_time;

            //copy_to_device
            tmp_copy_gpu_time = 0.0;
            tmp_copy_gpu_time-=MPI_Wtime();
            if(to_gpu){
                ssg->sampled_sgs[i]->allocate_dev_array_async(ssg->cs->stream);
                ssg->sampled_sgs[i]->copy_data_to_device_async(ssg->cs->stream);
            }
            tmp_copy_gpu_time+=MPI_Wtime();
            copy_gpu_time+=tmp_copy_gpu_time;
        }

        tmp_all_time+=MPI_Wtime();
        all_time+=tmp_all_time;
        auto actual_skip_batch_size = std::min((VertexId)skip_batch_size,work_range[1]-work_offset);
        work_offset+=actual_skip_batch_size;
        return ssg;
    }


    ~FastSampler(){
        ssg->~SampledSubgraph();
    }

    bool sample_not_finished(){
        return work_offset<work_range[1];
    }
    void restart(){
        work_offset=work_range[0];
    }



};
#endif