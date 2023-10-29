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
#ifndef NTSDATALODOR_HPP
#define NTSDATALODOR_HPP
#include <assert.h>
#include <map>
#include <math.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <sstream>
#include <execution>
#include <random>
#include "core/graph.hpp"
#include <tbb/concurrent_queue.h>

struct CacheVars{
    VertexId* cache_map;
    VertexId* cache_location;
    VertexId* dev_cache_map;
    VertexId* dev_cache_location;
    CacheVars(VertexId* cache_map_, VertexId* cache_location_, VertexId* dev_cache_map_, VertexId* dev_cache_location_):
        cache_map(cache_map_), cache_location(cache_location_), dev_cache_map(dev_cache_map_), dev_cache_location(dev_cache_location_){}
};

struct MultiCacheVars{
    VertexId* cache_map;
    VertexId* cache_location;
    VertexId** multi_dev_cache_map;
    VertexId** multi_dev_cache_location;
    int device_num;
    MultiCacheVars(VertexId* cache_map_, VertexId* cache_location_, VertexId** multi_dev_cache_map_,
                   VertexId** multi_dev_cache_location_, int device_num_): cache_map(cache_map_),
                   cache_location(cache_location_), multi_dev_cache_map(multi_dev_cache_map_),
                   multi_dev_cache_location(multi_dev_cache_location_), device_num(device_num_){}
};

struct NNVars{
    ValueType* shared_aggr;
    ValueType* shared_embedding;
    ValueType* dev_shared_aggr;
    ValueType* dev_shared_embedding;
    size_t vertex_num;
    int feature_len;
    int embedding_len;
    cudaEvent_t transfer_event;
    bool transfer_aggr;
    NNVars(ValueType* shared_aggr_, ValueType* shared_embedding_, size_t vertex_num_, int feature_len_, int embedding_len_,
           bool transfer_aggr_ = true)
        :shared_aggr(shared_aggr_), shared_embedding(shared_embedding_), vertex_num(vertex_num_), feature_len(feature_len_),
        embedding_len(embedding_len_), transfer_aggr(transfer_aggr_){
//        dev_shared_aggr = (ValueType*)getDevicePointer(shared_aggr);
//        dev_shared_embedding = (ValueType*) getDevicePointer(shared_embedding);
        if(transfer_aggr_) {
            cudaMalloc(&dev_shared_aggr, sizeof(ValueType) * vertex_num * feature_len);
        }
        cudaMalloc(&dev_shared_embedding, sizeof(ValueType) * vertex_num * embedding_len);
    }
    NNVars(ValueType* shared_embedding_, size_t vertex_num_, int embedding_len_, cudaStream_t cudaStream=0)
            :shared_embedding(shared_embedding_), vertex_num(vertex_num_),
             embedding_len(embedding_len_){
//        dev_shared_aggr = (ValueType*)getDevicePointer(shared_aggr);
//        dev_shared_embedding = (ValueType*) getDevicePointer(shared_embedding);
        transfer_aggr = false;
        cudaMallocAsync(&dev_shared_embedding, sizeof(ValueType) * vertex_num * embedding_len, cudaStream);
        std::printf("cache size: %u byte\n", sizeof(ValueType) * vertex_num * embedding_len);
//        cudaMalloc(&dev_shared_embedding, sizeof(ValueType) * vertex_num * embedding_len);
    }
};

class GNNDatum {
public:
  GNNContext *gnnctx;
  Graph<Empty> *graph;
  ValueType *local_feature; // features of local partition
  ValueType *dev_local_feature;
    ValueType **dev_local_feature_multi;

  ValueType *local_embedding; // embedding of local partition
  ValueType *dev_local_embedding;
 
  VertexId *CacheFlag; //0:未完成聚合  1：CPU已完成聚合 2：CPU的聚合已经传到GPU去
  VertexId *dev_CacheFlag;

  VertexId *CacheMap; // 存储缓存节点在cache中的下标
  VertexId *dev_CacheMap;

  // Note: 预算batch版本的变量
    std::vector<CacheVars*> cache_vars_vector;
    std::vector<NNVars*> nn_vars_vector;
    // 下面是Multi_GPU的版本
    // 下面是共用一份，所以算一份就行
    std::vector<MultiCacheVars*> multi_cache_vars_vector;
    // nnVars是每个GPU一个，然后还要使用其中的变量进行AllGather
    std::vector<std::vector<NNVars*>> multi_nn_vars_vector;
    // 用于标记cache_var到哪个batch的数组
    std::vector<int> current_epoch;
    std::vector<int> reuse_cache_var_countdown;
    std::mutex reuse_cache_mutex;
    std::mutex cache_var_mutex;

//    tbb::concurrent_queue<VertexId*> cache_map_queue;
//    tbb::concurrent_queue<VertexId*> cache_location_queue;
//    tbb::concurrent_queue<VertexId*> dev_cache_map_queue;
//    tbb::concurrent_queue<VertexId*> dev_cache_location_queue;
    tbb::concurrent_queue<CacheVars*> cache_var_queue;
    tbb::concurrent_queue<MultiCacheVars*> multi_cache_var_queue;

    // super-batch类型的PD-cache
    tbb::concurrent_queue<NNVars*> shared_nn_var_queue;
    tbb::concurrent_queue<NNVars*>  reuse_nn_var_queue;
    // Multi GPU类型的cache
    std::vector<tbb::concurrent_queue<NNVars*>> multi_shared_nn_queue;
    std::vector<tbb::concurrent_queue<NNVars*>> multi_reuse_nn_queue;


  ValueType *local_aggregation;
  ValueType *dev_local_aggregation;

    ValueType *local_aggregation_cpu;
    ValueType *dev_local_aggregation_cpu;
    ValueType *local_embedding_cpu; // embedding of local partition
    ValueType *dev_local_embedding_cpu;
    std::atomic<bool> gpu_flag;
    std::atomic<bool> cpu_flag;
    std::mutex share_mutex;
    std::condition_variable share_cv;
    // TODO: CPU端可能要上锁，GPU端使用自旋的方式，GPU端调用前要上锁，分别由两个函数控制（一个上锁，一个释放锁并唤醒CPU线程），CPU上锁就在交换函数里面控制就行

  VertexId *X_version;          // 存储CPU上的版本信息
  VertexId *dev_X_version;

  VertexId *Y_version;          // 存储GPU上的版本信息
  VertexId *dev_Y_version;

  ValueType *dev_share_embedding; //share computation
  ValueType *dev_share_aggregate;

  uint8_t *dev_mask_tensor;
  NtsVar mask_tensor;
  NtsVar aggregate_tensor;

  VertexId cache_num;
  long *local_label;        // labels of local partition
  int *local_mask;
  long *dev_local_label;
  long **dev_local_label_multi;
  int gpu_num;
  int max_threads;

  NCCL_Communicator* nccl_communicator;

   // mask(indicate whether data is for train, eval or test) of
                   // local partition

  // GNN datum world

// train:    0
// val:     1
// test:     2
    /**
     * @brief Construct a new GNNDatum::GNNDatum object.
     * initialize GNN Data using GNNContext and Graph.
     * Allocating space to save data. e.g. local feature, local label.
     * @param _gnnctx pointer to GNN Context
     * @param graph_ pointer to Graph
     */
    GNNDatum(GNNContext *_gnnctx, Graph<Empty> *graph_):gnnctx(_gnnctx),graph(graph_) {
//      gnnctx = _gnnctx;
//        graph = graph_;
      //local_feature = new ValueType[gnnctx->l_v_num * gnnctx->layer_size[0]];
      local_feature=(ValueType*)cudaMallocPinned((long)(gnnctx->l_v_num)*(gnnctx->layer_size[0])*sizeof(ValueType));
      // local_label = (long*)cudaMallocPinned((long)(gnnctx->l_v_num) * sizeof(long));
      local_label = new long[gnnctx->l_v_num];
      local_mask = new int[gnnctx->l_v_num];
      memset(local_mask, 1, sizeof(int) * gnnctx->l_v_num);
      max_threads = std::thread::hardware_concurrency();
    }

    void init_multi_gpu(int gpu_num, NCCL_Communicator* comms) {
        this->gpu_num = gpu_num;
        dev_local_feature_multi = new ValueType *[gpu_num];
        dev_local_label_multi = new long *[gpu_num];
        nccl_communicator = comms;
    }

    void init_cache_var(float cache_rate){
        cache_num = gnnctx->l_v_num * cache_rate;
        local_embedding = (ValueType*)cudaMallocPinned((long)(cache_num)*(gnnctx->layer_size[1])*sizeof(ValueType));
        CacheFlag = (VertexId*) cudaMallocPinned((long)(gnnctx->l_v_num) * sizeof(VertexId));
        CacheMap = (VertexId*) cudaMallocPinned((long)(gnnctx->l_v_num) * sizeof(VertexId));

        local_aggregation = (ValueType*) cudaMallocPinned(cache_num * (gnnctx->layer_size[0]) * sizeof(ValueType));
        X_version = (VertexId*) cudaMallocPinned(cache_num * sizeof(VertexId));
        Y_version = (VertexId*) cudaMallocPinned(cache_num * sizeof(VertexId));
        memset(X_version, 0, sizeof(VertexId) * cache_num);
        memset(Y_version, 0, sizeof(VertexId) * cache_num);


        assert(gnnctx->layer_size.size() > 1);
        dev_share_embedding = (ValueType*)cudaMallocGPU(cache_num * gnnctx->layer_size[1] * sizeof(ValueType));
//        aggregate_tensor = torch::zeros({cache_num, gnnctx->layer_size[0]},
//                                        at::TensorOptions().dtype(torch::kFloat32).device_index(0));
//        dev_share_aggregate = aggregate_tensor.packed_accessor<float, 2>().data();
//        mask_tensor = torch::zeros({cache_num, 1}, at::TensorOptions().dtype(torch::kBool).device_index(0));
//        dev_mask_tensor = mask_tensor.packed_accessor<uint8_t, 2>().data();
        dev_share_aggregate = (ValueType*)cudaMallocGPU(cache_num * gnnctx->layer_size[0] * sizeof(ValueType));
        //初始化和刷新
//        dev_share_grad = (ValueType*)cudaMallocGPU(cache_num * gnnctx->layer_size[gnnctx->max_layer - 1] * sizeof(ValueType));
        dev_CacheFlag = (VertexId*) getDevicePointer(CacheFlag);
        dev_CacheMap = (VertexId*) getDevicePointer(CacheMap);
        dev_local_embedding=(ValueType*)getDevicePointer(local_embedding);

        dev_local_aggregation = (ValueType*) getDevicePointer(local_aggregation);
        dev_X_version = (VertexId*) getDevicePointer(X_version);
        dev_Y_version = (VertexId*) getDevicePointer(Y_version);

        // 下面是使用多个缓冲区的做法
        local_aggregation_cpu = (ValueType*) cudaMallocPinned(cache_num * (gnnctx->layer_size[0]) * sizeof(ValueType));
        dev_local_aggregation_cpu = (ValueType*)getDevicePointer(local_aggregation_cpu);
        local_embedding_cpu = (ValueType*) cudaMallocPinned((long)(cache_num)*(gnnctx->layer_size[1])*sizeof(ValueType));
        dev_local_embedding_cpu = (ValueType*)getDevicePointer(local_embedding_cpu);
        cpu_flag.store(false);
        gpu_flag.store(false);

    }

    void init_super_batch_var(VertexId epoch_super_batch_num) {
        cache_vars_vector.resize(epoch_super_batch_num);
        nn_vars_vector.resize(epoch_super_batch_num);

        auto* cache_map = (VertexId*) cudaMallocPinned((long)(gnnctx->l_v_num) * sizeof(VertexId));
        auto* cache_location = (VertexId*) cudaMallocPinned((long)(gnnctx->l_v_num) * sizeof(VertexId));
        auto* dev_cache_map = (VertexId*) getDevicePointer(cache_map);
        auto* dev_cache_location = (VertexId*) getDevicePointer(cache_location);
        for(VertexId i = 0; i < gnnctx->l_v_num; i++){
            cache_map[i] = -1;
        }

        cache_var_queue.push(new CacheVars(cache_map, cache_location, dev_cache_map, dev_cache_location));
    }

    // 初始化多GPU中的每个的cache
    void init_multi_cache_var(VertexId epoch_super_batch_num, int device_num) {
        multi_cache_vars_vector.resize(epoch_super_batch_num);
        multi_nn_vars_vector.resize(device_num);
        for(int i = 0; i < device_num; i++) {
            multi_nn_vars_vector[i].resize(epoch_super_batch_num);
        }
        auto* cache_map = (VertexId*) cudaMallocPinned((long)(gnnctx->l_v_num) * sizeof(VertexId));
        auto* cache_location = (VertexId*) cudaMallocPinned((long)(gnnctx->l_v_num) * sizeof(VertexId));
        auto** multi_dev_cache_map = new VertexId*[device_num];
        auto** multi_dev_cache_location = new VertexId*[device_num];
        memset(cache_map, -1, (long)(gnnctx->l_v_num) * sizeof(VertexId));
        for(int i = 0; i < device_num; i++) {
            cudaSetUsingDevice(i);
            multi_dev_cache_map[i] = (VertexId*)getDevicePointer(cache_map);
            multi_dev_cache_location[i] = (VertexId*) getDevicePointer(cache_location);
        }
        multi_reuse_nn_queue.resize(device_num);
        multi_shared_nn_queue.resize(device_num);
        current_epoch.resize(epoch_super_batch_num);
        memset(current_epoch.data(), -1, current_epoch.size() * sizeof(int));
        reuse_cache_var_countdown.resize(epoch_super_batch_num);
        multi_cache_var_queue.push(new MultiCacheVars(cache_map, cache_location,
 multi_dev_cache_map, multi_dev_cache_location, device_num));
    }

    // 获取cache标记相关的变量，用于记录cache点
    CacheVars* new_cache_var(VertexId super_batch_id){
        CacheVars* cacheVars;
        if(!cache_var_queue.try_pop(cacheVars)){
            std::printf("创建了新的CacheVar\n");
            auto* cache_map = (VertexId*) cudaMallocPinned((long)(gnnctx->l_v_num) * sizeof(VertexId));
            auto* cache_location = (VertexId*) cudaMallocPinned((long)(gnnctx->l_v_num) * sizeof(VertexId));
            auto* dev_cache_map = (VertexId*) getDevicePointer(cache_map);
            auto* dev_cache_location = (VertexId*) getDevicePointer(cache_location);
            cacheVars = new CacheVars(cache_map, cache_location, dev_cache_map, dev_cache_location);
            for(VertexId i = 0; i < gnnctx->l_v_num; i++){
                cacheVars->cache_map[i] = -1;
            }
        }
        cache_vars_vector[super_batch_id] = cacheVars;
        return cacheVars;
    }

    // 用于获取多线程（多GPU）获取cache variable
    MultiCacheVars* multi_new_cache_var(VertexId super_batch_id, int epoch) {
        if(current_epoch[super_batch_id] != epoch) {
            std::unique_lock<std::mutex> cache_lock(cache_var_mutex, std::defer_lock);
            cache_lock.lock();

            if(current_epoch[super_batch_id] != epoch) {
                MultiCacheVars* multiCacheVars;
                if(!multi_cache_var_queue.try_pop(multiCacheVars)){

                    auto* cache_map = (VertexId*) cudaMallocPinnedMulti((long)(gnnctx->l_v_num) * sizeof(VertexId));
                    auto* cache_location = (VertexId*) cudaMallocPinnedMulti((long)(gnnctx->l_v_num) * sizeof(VertexId));
                    auto** multi_dev_cache_map = new VertexId*[gpu_num];
                    auto** multi_dev_cache_location = new VertexId*[gpu_num];
                    memset(cache_map, -1, (long)(gnnctx->l_v_num) * sizeof(VertexId));
                    for(int i = 0; i < gpu_num; i++) {
                        cudaSetUsingDevice(i);
                        multi_dev_cache_map[i] = (VertexId*)getDevicePointer(cache_map);
                        multi_dev_cache_location[i] = (VertexId*) getDevicePointer(cache_location);

                    }

//                    VertexId* cache_map, *cache_location;
//                    auto** multi_dev_cache_map = new VertexId*[gpu_num];
//                    auto** multi_dev_cache_location = new VertexId*[gpu_num];
//                    cache_map = (VertexId*)cudaMallocZero((long)(gnnctx->l_v_num) * sizeof(VertexId));
//                    cache_location = (VertexId*) cudaMallocZero((long)(gnnctx->l_v_num) * sizeof(VertexId));
//                    for(int i = 0; i < gpu_num; i++) {
//                        multi_dev_cache_map[i] = cache_map;
//                        multi_dev_cache_location[i] = cache_location;
//                    }

                    multiCacheVars = new MultiCacheVars(cache_map, cache_location, multi_dev_cache_map,
                                                        multi_dev_cache_location, gpu_num);
                }
                multi_cache_vars_vector[super_batch_id] = multiCacheVars;
                current_epoch[super_batch_id] = epoch;
                reuse_cache_var_countdown[super_batch_id] = gpu_num;
            }
            cache_lock.unlock();
        }
        return multi_cache_vars_vector[super_batch_id];
    }

    // 用于获取多线程（多GPU）获取cache variable
    MultiCacheVars* multi_new_cache_var(VertexId super_batch_id, int epoch, cudaStream_t cudaStream) {
        if(current_epoch[super_batch_id] != epoch) {
            std::unique_lock<std::mutex> cache_lock(cache_var_mutex, std::defer_lock);
            cache_lock.lock();

            if(current_epoch[super_batch_id] != epoch) {
                MultiCacheVars* multiCacheVars;
                if(!multi_cache_var_queue.try_pop(multiCacheVars)){

                    auto* cache_map = (VertexId*) cudaMallocPinnedMulti((long)(gnnctx->l_v_num) * sizeof(VertexId));
                    auto* cache_location = (VertexId*) cudaMallocPinnedMulti((long)(gnnctx->l_v_num) * sizeof(VertexId));
                    auto** multi_dev_cache_map = new VertexId*[gpu_num];
                    auto** multi_dev_cache_location = new VertexId*[gpu_num];
                    memset(cache_map, -1, (long)(gnnctx->l_v_num) * sizeof(VertexId));
                    for(int i = 0; i < gpu_num; i++) {
                        cudaSetUsingDevice(i);
//                        multi_dev_cache_map[i] = (VertexId*)getDevicePointer(cache_map);
//                        multi_dev_cache_location[i] = (VertexId*) getDevicePointer(cache_location);

                        allocate_gpu_edge_async(&multi_dev_cache_map[i], (long)(gnnctx->l_v_num), cudaStream);
                        allocate_gpu_edge_async(&multi_dev_cache_location[i], (long)(gnnctx->l_v_num), cudaStream);
                    }

//                    VertexId* cache_map, *cache_location;
//                    auto** multi_dev_cache_map = new VertexId*[gpu_num];
//                    auto** multi_dev_cache_location = new VertexId*[gpu_num];
//                    cache_map = (VertexId*)cudaMallocZero((long)(gnnctx->l_v_num) * sizeof(VertexId));
//                    cache_location = (VertexId*) cudaMallocZero((long)(gnnctx->l_v_num) * sizeof(VertexId));
//                    for(int i = 0; i < gpu_num; i++) {
//                        multi_dev_cache_map[i] = cache_map;
//                        multi_dev_cache_location[i] = cache_location;
//                    }

                    multiCacheVars = new MultiCacheVars(cache_map, cache_location, multi_dev_cache_map,
                                                        multi_dev_cache_location, gpu_num);
                }
                multi_cache_vars_vector[super_batch_id] = multiCacheVars;
                current_epoch[super_batch_id] = epoch;
                reuse_cache_var_countdown[super_batch_id] = gpu_num;
            }
            cache_lock.unlock();
        }
        return multi_cache_vars_vector[super_batch_id];
    }

    // 获取cache标记相关的变量，用于记录cache点
    CacheVars* get_cache_var(VertexId super_batch_id){
        return cache_vars_vector[super_batch_id];
    }

    // 获取cache标记相关的变量，用于记录cache点
    MultiCacheVars* get_multi_cache_var(VertexId super_batch_id){
        return multi_cache_vars_vector[super_batch_id];
    }

    NNVars* new_nn_var(VertexId super_batch_id) {
        NNVars* nnVars;
        // TODO: 这里可以优化成条件变量的形式
        while(!shared_nn_var_queue.try_pop(nnVars)) {
            std::this_thread::yield();
        }
        // 等待GPU传输完成
//        std::printf("获取nn_var完毕，开始检查是否传输完成\n");
        cudaEventSynchronize(nnVars->transfer_event);
        cudaEventDestroy(nnVars->transfer_event);
        nn_vars_vector[super_batch_id] = nnVars;
        return nnVars;
    }

    NNVars* multi_new_nn_var(VertexId super_batch_id, int device_id) {
        NNVars* nnVars;
        // TODO: 这里可以优化成条件变量的形式
        while(!multi_shared_nn_queue[device_id].try_pop(nnVars)) {
            std::this_thread::yield();
        }
        // 等待GPU传输完成
//        std::printf("获取nn_var完毕，开始检查是否传输完成\n");
        cudaEventSynchronize(nnVars->transfer_event);
        multi_nn_vars_vector[device_id][super_batch_id] = nnVars;
        return nnVars;
    }

    NNVars* get_nn_var(VertexId super_batch_id) {
        return nn_vars_vector[super_batch_id];
    }
    NNVars* get_multi_nn_var(VertexId super_batch_id, int device_num) {
        return multi_nn_vars_vector[device_num][super_batch_id];
    }

    void add_cache_var(CacheVars* cacheVars) {
        cache_var_queue.push(cacheVars);
    }

    void set_cache_index(VertexId* cache_map, VertexId* cache_location, VertexId super_batch_id, VertexId batch_cache_num,
                         std::vector<VertexId>& cache_ids){
        VertexId start = super_batch_id * batch_cache_num;
#pragma omp parallel for
        for(VertexId i = start; i < start + batch_cache_num; i++) {
//            if(i >= cache_ids.size()) {
//                std::printf("i: %u, cache ids size: %u, super batch id: %u\n", i , cache_ids.size(), super_batch_id);
//            }
            assert(cache_ids.size() > i);
            cache_map[cache_ids[i]] = super_batch_id;
            cache_location[cache_ids[i]] = i - start;
        }
    }


    void set_cache_index(VertexId* cache_map, VertexId* cache_location, VertexId super_batch_id, std::vector<VertexId>& batch_cache_num,
                         std::vector<VertexId>& cache_ids){
        VertexId start = 0;
        for(VertexId i = 0; i < super_batch_id; i++){
            start += batch_cache_num[i];
        }
        // std::printf("batch_cache_num[%d]: %d\n",super_batch_id , batch_cache_num[super_batch_id]);
        //VertexId start = super_batch_id * batch_cache_num;
//        std::printf("test start:%d batch_cache_num:%d \n",start, batch_cache_num[super_batch_id]);
#pragma omp parallel for
        for(VertexId i = start; i < start + batch_cache_num[super_batch_id]; i++) {
//            if(i >= cache_ids.size()) {
//                std::printf("i: %u, cache ids size: %u, super batch id: %u\n", i , cache_ids.size(), super_batch_id);
//            }
//            if(i >= cache_ids.size()) {
//                std::printf("i: %u, start: %u, batch cache: %u, super_batch_id: %d, cache_ids.size: %ld, batch_cache size: %ld\n",
//                            i, start, batch_cache_num[super_batch_id], super_batch_id, cache_ids.size(), batch_cache_num.size());
//            }
            assert(cache_ids.size() > i);
            cache_map[cache_ids[i]] = super_batch_id;
            cache_location[cache_ids[i]] = i - start;
        }
//        std::printf("test end 1\n");
    }


    void genereate_gpu_data(){
        dev_local_label=(long*)cudaMallocGPU(gnnctx->l_v_num*sizeof(long));
        dev_local_feature=(ValueType*)getDevicePointer(local_feature);
        move_bytes_in(dev_local_label,local_label,gnnctx->l_v_num*sizeof(long));
    }

    void generate_multi_gpu_data() {
        for(int i = 0; i < gpu_num; i++) {
            cudaSetUsingDevice(i);
            dev_local_label_multi[i] = (long*)cudaMallocGPU(gnnctx->l_v_num*sizeof(long));
            dev_local_feature_multi[i] = (ValueType*)getDevicePointer(local_feature);
            move_bytes_in(dev_local_label_multi[i],local_label,gnnctx->l_v_num*sizeof(long));
        }
    }

    void generate_aggregate_pd_data(){
        local_aggregation = (ValueType*) cudaMallocPinned(cache_num * (gnnctx->layer_size[0]) * sizeof(ValueType));
        dev_local_aggregation = (ValueType*) getDevicePointer(local_aggregation);
    }

void reset_cache_flag(std::vector<VertexId>& cache_ids){
#pragma omp parallel for
    for(int i = 0; i < cache_num; i++){
        CacheFlag[cache_ids[i]] = 0;
    }
}

void move_data_to_local_cache(VertexId vertex_num, int feature_len, ValueType* data,
                              VertexId* ids, uint8_t* masks, VertexId start){
//      std::printf("batch start: %d, batch num: %d\n", start, vertex_num);
#pragma omp parallel for
    for(VertexId i = 0; i < vertex_num; i++){
        if(CacheFlag[ids[i]] == 0) {
            auto index = CacheMap[ids[i]];  // 获取该点在cache中的第几行
            memcpy(&local_embedding[index * feature_len], &data[i * feature_len], sizeof(ValueType)*feature_len);
            if(__sync_bool_compare_and_swap(&(CacheFlag[ids[i]]), 0u, 1u)){
                masks[start + i] = 1;
            }
        }
    }
}

    void try_swap_buffer(){
        if(!gpu_flag.load()){
            cpu_flag.store(true);
            std::swap(local_embedding, local_embedding_cpu);
            std::swap(local_aggregation, local_aggregation_cpu);
            std::swap(X_version, Y_version);
            std::swap(dev_X_version, dev_Y_version);
            cpu_flag.store(false);
        }
    }


    ValueType* move_data_to_pin_memory(VertexId vertex_num, int feature_len, ValueType* feature){
        ValueType* pinned_memory = (ValueType*)cudaMallocPinned(sizeof(ValueType) * vertex_num * feature_len);
        memcpy(pinned_memory, feature, vertex_num * feature_len * sizeof(ValueType));
        return pinned_memory;
    }


    void move_data_to_pin_memory(VertexId vertex_num, int feature_len, ValueType* feature, ValueType* pinned_memory){
        memcpy(pinned_memory, feature, vertex_num * feature_len * sizeof(ValueType));
    }

    void move_data_to_pin_memory(VertexId vertex_num, int feature_len, int embedding_len, ValueType* feature,
                                 ValueType* embedding) {
        NNVars* nnVars;
        if(!reuse_nn_var_queue.try_pop(nnVars)) {
            ValueType* shared_aggr = (ValueType*)cudaMallocPinned(sizeof(ValueType) * vertex_num * feature_len);
            ValueType* shared_embedding = (ValueType*)cudaMallocPinned(sizeof(ValueType) * vertex_num * embedding_len);
            nnVars = new NNVars(shared_aggr, shared_embedding, vertex_num, feature_len, embedding_len);
        }
        memcpy(nnVars->shared_aggr, feature, vertex_num * feature_len * sizeof(ValueType));
        memcpy(nnVars->shared_embedding, embedding, vertex_num * embedding_len * sizeof(ValueType));
        shared_nn_var_queue.push(nnVars);
    }


    void move_data_to_pin_memory(VertexId vertex_num, int feature_len, int embedding_len, ValueType* feature,
                                 ValueType* embedding, cudaStream_t cudaStream) {
        NNVars* nnVars;
        if(!reuse_nn_var_queue.try_pop(nnVars)) {
            std::printf("CPU创建了新的nnVars\n");
            ValueType* shared_aggr = (ValueType*)cudaMallocPinned(sizeof(ValueType) * vertex_num * feature_len);
            ValueType* shared_embedding = (ValueType*)cudaMallocPinned(sizeof(ValueType) * vertex_num * embedding_len);
            nnVars = new NNVars(shared_aggr, shared_embedding, vertex_num, feature_len, embedding_len);
        }
        memcpy(nnVars->shared_aggr, feature, vertex_num * feature_len * sizeof(ValueType));
        memcpy(nnVars->shared_embedding, embedding, vertex_num * embedding_len * sizeof(ValueType));
        move_nn_data_to_gpu(nnVars, cudaStream);
        shared_nn_var_queue.push(nnVars);
    }

    void move_embedding_to_pin_memory(VertexId vertex_num, int feature_len, int embedding_len, ValueType* feature,
                                      ValueType* embedding, cudaStream_t cudaStream) {
        NNVars* nnVars;
        if(!reuse_nn_var_queue.try_pop(nnVars)) {
            std::printf("CPU创建了新的nnVars\n");
            ValueType* shared_aggr = nullptr;
            ValueType* shared_embedding = (ValueType*)cudaMallocPinned(sizeof(ValueType) * vertex_num * embedding_len);
            nnVars = new NNVars(shared_aggr, shared_embedding, vertex_num, feature_len, embedding_len, false);
        }
//        memcpy(nnVars->shared_aggr, feature, vertex_num * feature_len * sizeof(ValueType));
        memcpy(nnVars->shared_embedding, embedding, vertex_num * embedding_len * sizeof(ValueType));
        move_nn_data_to_gpu(nnVars, cudaStream);
        shared_nn_var_queue.push(nnVars);

    }

    void move_embedding_to_gpu(VertexId vertex_num, int embedding_len, ValueType *embedding, cudaStream_t cudaStream) {
        NNVars* nnVars;
        if(!reuse_nn_var_queue.try_pop(nnVars)) {
            std::printf("CPU创建了新的nnVars\n");
            ValueType* shared_embedding = (ValueType*)cudaMallocPinned(sizeof(ValueType) * vertex_num * embedding_len);
            nnVars = new NNVars(shared_embedding, vertex_num, embedding_len);
        }
//        memcpy(nnVars->shared_aggr, feature, vertex_num * feature_len * sizeof(ValueType));
        memcpy(nnVars->shared_embedding, embedding, vertex_num * embedding_len * sizeof(ValueType));
        move_nn_data_to_gpu(nnVars, cudaStream);
        shared_nn_var_queue.push(nnVars);
    }

    void move_embedding_to_gpu(VertexId top_num, VertexId vertex_num, int embedding_len, ValueType *embedding, cudaStream_t cudaStream) {
        NNVars* nnVars;
        if(!reuse_nn_var_queue.try_pop(nnVars)) {
            std::printf("CPU创建了新的nnVars\n");
            ValueType* shared_embedding = (ValueType*)cudaMallocPinned(sizeof(ValueType) * top_num * embedding_len);
            nnVars = new NNVars(shared_embedding, top_num, embedding_len);
        }
//        memcpy(nnVars->shared_aggr, feature, vertex_num * feature_len * sizeof(ValueType));
        memcpy(nnVars->shared_embedding, embedding, vertex_num * embedding_len * sizeof(ValueType)); //耗时，最好直接修改pinnmemory 上的share embedding
        move_nn_data_to_gpu(nnVars, vertex_num, cudaStream); //只传对应batchnum的嵌入
        shared_nn_var_queue.push(nnVars);
    }

    // TODO: 应该是每个device一个shared nn结构，即每回都从不同的队列中取cache数据
    void move_embedding_to_multi_gpu(VertexId vertex_num, int embedding_len, ValueType *embedding,
                                     cudaStream_t* cudaStream, int device_num, std::vector<VertexId>& dev_cache_offset){
        // TODO: 这里应该定的是一个偏移量
        NNVars* nnVars;
        ValueType* shared_embedding = (ValueType*)cudaMallocPinned(sizeof(ValueType) * vertex_num * embedding_len);
        memcpy(shared_embedding, embedding, vertex_num * embedding_len * sizeof(ValueType));
        std::vector<NNVars*> nnvarsVector;
        for(int i = 0; i < device_num; i++) {
            // TODO: CPU的数据应该是只要一份就行
            nnVars = new NNVars(shared_embedding, vertex_num, embedding_len,
                                cudaStream[i]);
            nnvarsVector.emplace_back(nnVars);
        }
        // TODO: 下面是将数据发送到GPU
         move_nn_data_to_multi_gpu(nnvarsVector, dev_cache_offset, cudaStream, device_num);
        for(int i = 0; i < device_num; i++) {
            multi_shared_nn_queue[i].push(nnvarsVector[i]);
        }
    }

    void move_embedding_to_multi_gpu(VertexId top_num, VertexId vertex_num, int embedding_len, ValueType *embedding,
                                     cudaStream_t* cudaStream, int device_num, std::vector<VertexId>& dev_cache_offset){
        // TODO: 这里应该定的是一个偏移量
        ValueType* shared_embedding = (ValueType*)cudaMallocPinned(sizeof(ValueType) * vertex_num * embedding_len);
        memcpy(shared_embedding, embedding, vertex_num * embedding_len * sizeof(ValueType));
        std::vector<NNVars*> nnvarsVector;
        for(int i = 0; i < device_num; i++) {
            // TODO: CPU的数据应该是只要一份就行
            NNVars* nnVars;
            if(!multi_reuse_nn_queue[i].try_pop(nnVars)){
                nnVars = new NNVars(shared_embedding, top_num, embedding_len,
                                    cudaStream[i]);
            }
            nnvarsVector.emplace_back(nnVars);
        }
//        std::printf("vertex num: %d\n", vertex_num);
//        for(int i = 0; i < dev_cache_offset.size(); i++) {
//            std::printf("%d ", dev_cache_offset[i]);
//        }
//        std::printf("\n");
        // TODO: 下面是将数据发送到GPU
        move_nn_data_to_multi_gpu(nnvarsVector, dev_cache_offset, cudaStream, device_num);
        for(int i = 0; i < device_num; i++) {
            multi_shared_nn_queue[i].push(nnvarsVector[i]);
        }
    }

    void move_gpu_data_to_cache_memory(VertexId vertex_num, int feature_len, int embedding_len, ValueType* feature,
                                   ValueType* embedding, cudaStream_t cudaStream){
        NNVars* nnVars;
        if(!reuse_nn_var_queue.try_pop(nnVars)) {
            std::printf("CPU创建了新的nnVars\n");
            ValueType* shared_aggr = nullptr;
            ValueType* shared_embedding = nullptr;
            nnVars = new NNVars(shared_aggr, shared_embedding, vertex_num, feature_len, embedding_len);
        }

        cudaEvent_t transfer_event;
        cudaEventCreate(&transfer_event);

        cudaMemcpyAsync(nnVars->dev_shared_aggr, feature, sizeof(ValueType) * nnVars->vertex_num * nnVars->feature_len,
                        cudaMemcpyHostToDevice, cudaStream);
        cudaMemcpyAsync(nnVars->dev_shared_embedding, embedding, sizeof(ValueType) * nnVars->vertex_num * nnVars->embedding_len,
                        cudaMemcpyHostToDevice, cudaStream);
        cudaEventRecord(transfer_event, cudaStream);
        nnVars->transfer_event = transfer_event;
        shared_nn_var_queue.push(nnVars);
    }

    void recycle_memory(CacheVars* cacheVars, NNVars* nnVars) {
        cache_var_queue.push(cacheVars);
        reuse_nn_var_queue.push(nnVars);
    }

    void recycle_multi_gpu_memory(MultiCacheVars* cacheVars, NNVars* nnVars,
                                  VertexId super_batch_id, int device_id) {
        write_add(&reuse_cache_var_countdown[super_batch_id], -1);
        if(reuse_cache_var_countdown[super_batch_id] == 0){
            std::unique_lock<std::mutex> cache_lock(reuse_cache_mutex, std::defer_lock);
            cache_lock.lock();
            if(reuse_cache_var_countdown[super_batch_id] != -1) {
                multi_cache_var_queue.push(cacheVars);
                reuse_cache_var_countdown[super_batch_id] = -1;
            }
            cache_lock.unlock();
        }
        multi_reuse_nn_queue[device_id].push(nnVars);

    }

    void move_nn_data_to_gpu(NNVars* nnVars, cudaStream_t cudaStream = 0) {
        cudaEvent_t transfer_event;
        cudaEventCreate(&transfer_event);
        if(nnVars->transfer_aggr) {
            cudaMemcpyAsync(nnVars->dev_shared_aggr, nnVars->shared_aggr, sizeof(ValueType) * nnVars->vertex_num * nnVars->feature_len,
                            cudaMemcpyHostToDevice, cudaStream);
        }
        cudaMemcpyAsync(nnVars->dev_shared_embedding, nnVars->shared_embedding, sizeof(ValueType) * nnVars->vertex_num * nnVars->embedding_len,
                        cudaMemcpyHostToDevice, cudaStream);
        cudaEventRecord(transfer_event, cudaStream);
        nnVars->transfer_event = transfer_event;
    }

    void move_nn_data_to_gpu(NNVars* nnVars,VertexId vertex_num, cudaStream_t cudaStream = 0) {
        cudaEvent_t transfer_event;
        cudaEventCreate(&transfer_event);
        if(nnVars->transfer_aggr) {
            cudaMemcpyAsync(nnVars->dev_shared_aggr, nnVars->shared_aggr, sizeof(ValueType) * vertex_num * nnVars->feature_len,
                            cudaMemcpyHostToDevice, cudaStream);
        }
        cudaMemcpyAsync(nnVars->dev_shared_embedding, nnVars->shared_embedding, sizeof(ValueType) * vertex_num * nnVars->embedding_len,
                        cudaMemcpyHostToDevice, cudaStream);
        cudaEventRecord(transfer_event, cudaStream);
        nnVars->transfer_event = transfer_event;
    }


    void move_nn_data_to_multi_gpu(std::vector<NNVars*>& nnVars, std::vector<VertexId>& dev_cache_offset,
                                   cudaStream_t* cudaStream = 0, int device_num = 1) {
        // TODO: 数据传输完成之后应该将event销毁
        // TODO: 缓存的顶点数量应该是设备数量的倍数
        // TODO: 如果失败的话可以尝试用多线程进行通信
        std::vector<std::thread> gather_threads;
        for(int i = 0; i < device_num; i++) {
            gather_threads.emplace_back([&](int dev_id){
                cudaSetUsingDevice(dev_id);
                cudaEvent_t transfer_event;
                cudaEventCreate(&transfer_event);

            //     assert(dev_cache_offset[device_num] <= nnVars[dev_id]->vertex_num);
            //     size_t transfer_size = dev_cache_offset[device_num] *
            //             nnVars[dev_id]->embedding_len * sizeof(ValueType);
            //    cudaMemcpyAsync(&(nnVars[dev_id]->dev_shared_embedding[0]), &(nnVars[dev_id]->shared_embedding[0]),
            //                    transfer_size,cudaMemcpyHostToDevice, cudaStream[dev_id]);


                // 确定传输的数据量
                size_t transfer_size = (dev_cache_offset[dev_id+1] - dev_cache_offset[dev_id]) *
                        nnVars[dev_id]->embedding_len * sizeof(ValueType);
                size_t start = dev_cache_offset[dev_id];

               cudaMemcpyAsync(&(nnVars[dev_id]->dev_shared_embedding[start]), &(nnVars[dev_id]->shared_embedding[start]),
                               transfer_size,cudaMemcpyHostToDevice, cudaStream[dev_id]);
                // std::printf("offset %d: %u, offset last: %u\n", dev_id, dev_cache_offset[dev_id], dev_cache_offset[gpu_num]);
               nccl_communicator->AllGather(dev_id, nnVars[dev_id]->dev_shared_embedding, dev_cache_offset[1] - dev_cache_offset[0],
                                                cudaStream[dev_id]);

//                std::printf("after device %d call allGather\n", dev_id);
                cudaEventRecord(transfer_event, cudaStream[dev_id]);
                nnVars[dev_id]->transfer_event = transfer_event;

            }, i);
        }
        for(int i = 0; i < device_num; i++) {
            gather_threads[i].join();
        }

    }




    void set_gpu_transfer_flag(){
        // 自旋等待
        gpu_flag.store(true);
        while(cpu_flag.load()) std::this_thread::yield();
    }

    void unset_gpu_transfer_flag(){
        gpu_flag.store(false);
    }


void move_data_to_local_cache(VertexId vertex_num, int feature_len, int embedding_len, ValueType* feature,
                              ValueType* embedding, VertexId* ids, VertexId start, uint32_t version){
        if(vertex_num <= 0){
            return;
        }
        if(X_version[CacheMap[ids[0]]] >= version){
            return;
        }
//        std::printf("feature len: %d, embedding len: %d, version: %d\n", feature_len, embedding_len, version);
//#pragma omp parallel for num_threads(max_threads)
//        for(VertexId i = 0; i < vertex_num; i++) {
//            auto index = CacheMap[ids[i]];
//            X_version[index] = version;
////            __asm__ __volatile__ ("" : : : "memory");
//            memcpy(&local_aggregation_cpu[index * feature_len], &feature[i * feature_len], sizeof(ValueType)*feature_len);
//            memcpy(&local_embedding_cpu[index * embedding_len], &embedding[i * embedding_len], sizeof(ValueType)*embedding_len);
////            Y_version[index] = X_version[index];
////            __asm__ __volatile__ ("" : : : "memory");
//
//        }


        std::vector<VertexId> ids_vertor(ids, ids+vertex_num);
        assert(ids_vertor.size() == vertex_num);
        std::for_each(std::execution::par, ids_vertor.begin(), ids_vertor.end(), [&](VertexId const& glabol_id){
            VertexId i = &glabol_id - &ids_vertor[0];
            assert(i >= 0 && i < ids_vertor.size());
            auto index = CacheMap[glabol_id];
            X_version[index] = version;
//            __asm__ __volatile__ ("" : : : "memory");
//            std::copy(std::execution::par, &feature[i * feature_len],&feature[(i+1) * feature_len], &local_aggregation_cpu[index * feature_len]);
//            std::copy(std::execution::par, &embedding[i * embedding_len],&embedding[(i+1) * embedding_len], &local_embedding_cpu[index * embedding_len]);
            memcpy(&local_aggregation_cpu[index * feature_len], &feature[i * feature_len], sizeof(ValueType)*feature_len);
            memcpy(&local_embedding_cpu[index * embedding_len], &embedding[i * embedding_len], sizeof(ValueType)*embedding_len);

        });

    }


/**
 * @brief
 * generate random data for feature, label and mask
 */
void random_generate(float train_rate=0.65, float val_rate=0.10, float test_rate=0.25) {
    VertexId max_train_id = gnnctx->l_v_num * train_rate;
    VertexId max_val_id = gnnctx->l_v_num * val_rate + max_train_id;
//     int threads = omp_get_num_threads();
//     std::random_device rd;
//     std::mt19937 mts[threads];
//     for(int i = 0; i < threads; i++) {
//         mts[i] = std::mt19937(rd());
//     }

// #pragma omp parallel for
  for (uint32_t i = 0; i < gnnctx->l_v_num; i++) {
    for (int j = 0; j < gnnctx->layer_size[0]; j++) {
      uint64_t index = i * gnnctx->layer_size[0] + j;
      local_feature[index] = 1.0;
    }
    local_label[i] = (rand()) % gnnctx->label_num;
//    local_mask[i] = i % 3;
    if(i < max_train_id) {
        local_mask[i] = 0;
    } else if(i < max_val_id) {
        local_mask[i] = 1;
    } else {
        local_mask[i] = 2;
    }
  }
}

/**
 * @brief
 * generate random data for feature, label and mask
 */
    void read_mask_random_other(std::string mask_file) {
//        VertexId max_train_id = gnnctx->l_v_num * train_rate;
//        VertexId max_val_id = gnnctx->l_v_num * val_rate + max_train_id;
//     int threads = omp_get_num_threads();
//     std::random_device rd;
//     std::mt19937 mts[threads];
//     for(int i = 0; i < threads; i++) {
//         mts[i] = std::mt19937(rd());
//     }


        std::ifstream input_msk(mask_file.c_str(), std::ios::in);
        VertexId id;
        VertexId train_count = 0;

        while (input_msk >> id) {
            VertexId id_trans = id - gnnctx->p_v_s;
            if ((gnnctx->p_v_s <= id) && (gnnctx->p_v_e > id)) {
                std::string msk;
                input_msk >> msk;
                // std::cout<<la<<" "<<msk<<std::endl;
                if (msk.compare("train") == 0) {
                    local_mask[id_trans] = 0;
                    train_count++;
                } else if (msk.compare("eval") == 0 || msk.compare("val") == 0) {
                    local_mask[id_trans] = 1;
                } else if (msk.compare("test") == 0) {
                    local_mask[id_trans] = 2;
                } else {
                    local_mask[id_trans] = 3;
                }

            }
        }
        input_msk.close();

        std::printf("train 点数量: %u\n", train_count);

// #pragma omp parallel for
        for (uint32_t i = 0; i < gnnctx->l_v_num; i++) {
            for (int j = 0; j < gnnctx->layer_size[0]; j++) {
                uint64_t index = i * gnnctx->layer_size[0] + j;
                local_feature[index] = 1.0;
            }
            local_label[i] = (rand()) % gnnctx->label_num;
//    local_mask[i] = i % 3;
        }
    }

/**
 * @brief
 * Create tensor corresponding to local label
 * @param target target tensor where we should place local label
 */
void registLabel(NtsVar &target) {
  target = graph->Nts->NewLeafKLongTensor(local_label, {static_cast<long>(gnnctx->l_v_num)});
  // torch::from_blob(local_label, gnnctx->l_v_num, torch::kLong);
}

/**
 * @brief
 * Create tensor corresponding to local mask
 * @param mask target tensor where we should place local mask
 */
void registMask(NtsVar &mask) {
  mask = graph->Nts->NewLeafKIntTensor(local_mask, {static_cast<long>(gnnctx->l_v_num), 1});
  // torch::from_blob(local_mask, {gnnctx->l_v_num,1}, torch::kInt32);
}

/**
 * @brief
 * read feature and label from file.
 * file format should be  ID Feature * (feature size) Label
 * @param inputF path to input feature
 * @param inputL path to input label
 */
void readFtrFrom1(std::string inputF, std::string inputL) {

  std::string str;
  std::ifstream input_ftr(inputF.c_str(), std::ios::in);
  std::ifstream input_lbl(inputL.c_str(), std::ios::in);
  // std::ofstream outputl("cora.labeltable",std::ios::out);
  // ID    F   F   F   F   F   F   F   L
  if (!input_ftr.is_open()) {
    std::cout << "open feature file fail!" << std::endl;
    return;
  }
  if (!input_lbl.is_open()) {
    std::cout << "open label file fail!" << std::endl;
    return;
  }
  ValueType *con_tmp = new ValueType[gnnctx->layer_size[0]];
  // TODO: figure out what is la
  std::string la;
  // std::cout<<"finish1"<<std::endl;
  VertexId id = 0;
  while (input_ftr >> id) {
    // feature size
    VertexId size_0 = gnnctx->layer_size[0];
    // translate vertex id to local vertex id
    VertexId id_trans = id - gnnctx->p_v_s;
    if ((gnnctx->p_v_s <= id) && (gnnctx->p_v_e > id)) {
      // read feature
      for (int i = 0; i < size_0; i++) {
        input_ftr >> local_feature[size_0 * id_trans + i];
      }
      input_lbl >> la;
      // read label
      input_lbl >> local_label[id_trans];
      // partition data set based on id
      local_mask[id_trans] = id % 3;
    } else {
      // dump the data which doesn't belong to local partition
      for (int i = 0; i < size_0; i++) {
        input_ftr >> con_tmp[i];
      }
      input_lbl >> la;
      input_lbl >> la;
    }
  }
  delete[] con_tmp;
  input_ftr.close();
  input_lbl.close();
}

/**
 * @brief
 * read feature, label and mask from file.
 * @param inputF path to feature file
 * @param inputL path to label file
 * @param inputM path to mask file
 */
void readFeature_Label_Mask(std::string inputF, std::string inputL,
                                      std::string inputM) {

  // logic here is exactly the same as read feature and label from file
  std::string str;
  std::ifstream input_ftr(inputF.c_str(), std::ios::in);
  std::ifstream input_lbl(inputL.c_str(), std::ios::in);
  std::ifstream input_msk(inputM.c_str(), std::ios::in);
  // std::ofstream outputl("cora.labeltable",std::ios::out);
  // ID    F   F   F   F   F   F   F   L
  if (!input_ftr.is_open()) {
    std::cout << "open feature file fail!" << std::endl;
    return;
  }
  if (!input_lbl.is_open()) {
    std::cout << "open label file fail!" << std::endl;
    return;
  }
  if (!input_msk.is_open()) {
    std::cout << "open mask file fail!" << std::endl;
    return;
  }
  ValueType *con_tmp = new ValueType[gnnctx->layer_size[0]];
  std::string la;
  // std::cout<<"finish1"<<std::endl;
  VertexId id = 0;
  while (input_ftr >> id) {
    VertexId size_0 = gnnctx->layer_size[0];
    VertexId id_trans = id - gnnctx->p_v_s;
    if ((gnnctx->p_v_s <= id) && (gnnctx->p_v_e > id)) {
      for (int i = 0; i < size_0; i++) {
        input_ftr >> local_feature[size_0 * id_trans + i];
      }
      input_lbl >> la;
      input_lbl >> local_label[id_trans];

      input_msk >> la;
      std::string msk;
      input_msk >> msk;
      // std::cout<<la<<" "<<msk<<std::endl;
      if (msk.compare("train") == 0) {
        local_mask[id_trans] = 0;
      } else if (msk.compare("eval") == 0 || msk.compare("val") == 0) {
        local_mask[id_trans] = 1;
      } else if (msk.compare("test") == 0) {
        local_mask[id_trans] = 2;
      } else {
        local_mask[id_trans] = 3;
      }

    } else {
      for (int i = 0; i < size_0; i++) {
        input_ftr >> con_tmp[i];
      }

      input_lbl >> la;
      input_lbl >> la;

      input_msk >> la;
      input_msk >> la;
    }
  }
  delete[] con_tmp;
  input_ftr.close();
  input_lbl.close();
}

void readFeature_Label_Mask_OGB(std::string inputF, std::string inputL,
                                      std::string inputM) {

  // logic here is exactly the same as read feature and label from file
  std::string str;
  std::ifstream input_ftr(inputF.c_str(), std::ios::in);
  std::ifstream input_lbl(inputL.c_str(), std::ios::in);
  // ID    F   F   F   F   F   F   F   L
  std::cout<<inputF<<std::endl;
  if (!input_ftr.is_open()) {
    std::cout << "open feature file fail!" << std::endl;
    return;
  }
  if (!input_lbl.is_open()) {
    std::cout << "open label file fail!" << std::endl;
    return;
  }
  ValueType *con_tmp = new ValueType[gnnctx->layer_size[0]];
  std::string la;
  std::string featStr;
  for (VertexId id = 0;id<graph->vertices;id++) {
    VertexId size_0 = gnnctx->layer_size[0];
    VertexId id_trans = id - gnnctx->p_v_s;
    if ((gnnctx->p_v_s <= id) && (gnnctx->p_v_e > id)) {
        getline(input_ftr,featStr);
        std::stringstream ss(featStr);
        std::string feat_u;
        int i=0;
        while(getline(ss,feat_u,',')){
            local_feature[size_0 * id_trans + i]=std::atof(feat_u.c_str());
          //  if(id==0){
          //      std::cout<<std::atof(feat_u.c_str())<<std::endl;
          //  }
            i++;
        }assert(i==size_0);       
      //input_lbl >> la;
      input_lbl >> local_label[id_trans];

    } else {
      getline(input_ftr,featStr);
      input_lbl >> la;
    }
  }
  
  std::string inputM_train=inputM;
  inputM_train.append("/train.csv");
  std::string inputM_val=inputM;
  inputM_val.append("/valid.csv");
  std::string inputM_test=inputM;
  inputM_test.append("/test.csv");
  std::ifstream input_msk_train(inputM_train.c_str(), std::ios::in);
  if (!input_msk_train.is_open()) {
    std::cout << "open input_msk_train file fail!" << std::endl;
    return;
  }
  std::ifstream input_msk_val(inputM_val.c_str(), std::ios::in);
  if (!input_msk_val.is_open()) {
    std::cout <<inputM_val<< "open input_msk_val file fail!" << std::endl;
    return;
  }
  std::ifstream input_msk_test(inputM_test.c_str(), std::ios::in);
  if (!input_msk_test.is_open()) {
    std::cout << "open input_msk_test file fail!" << std::endl;
    return;
  }
  VertexId vtx=0;
  while(input_msk_train>>vtx){//train
      local_mask[vtx] = 0;
  }
  while(input_msk_val>>vtx){//val
      local_mask[vtx] = 1;
  }
  while(input_msk_test>>vtx){//test
      local_mask[vtx] = 2;
  }
  
  
  delete[] con_tmp;
  input_ftr.close();
  input_lbl.close();
}

};

#endif
