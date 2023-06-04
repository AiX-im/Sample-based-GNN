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

class GNNDatum {
public:
  GNNContext *gnnctx;
  Graph<Empty> *graph;
  ValueType *local_feature; // features of local partition
  ValueType *dev_local_feature;

  ValueType *local_embedding; // embedding of local partition
  ValueType *dev_local_embedding;
 
  VertexId *CacheFlag; //0:未完成聚合  1：CPU已完成聚合 2：CPU的聚合已经传到GPU去
  VertexId *dev_CacheFlag;

  VertexId *CacheMap; // 存储缓存节点在cache中的下标
  VertexId *dev_CacheMap;

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
  int max_threads;
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



    void genereate_gpu_data(){
        dev_local_label=(long*)cudaMallocGPU(gnnctx->l_v_num*sizeof(long));
        dev_local_feature=(ValueType*)getDevicePointer(local_feature);
        move_bytes_in(dev_local_label,local_label,gnnctx->l_v_num*sizeof(long));
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
