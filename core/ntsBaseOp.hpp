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
#ifndef NTSBASEOP_HPP
#define NTSBASEOP_HPP
#include "core/graph.hpp"
#include "core/PartitionedGraph.hpp"
#include "FullyRepGraph.hpp"
#include <immintrin.h>

#define print_tensor_size(tensor) (nts::op::print_NtsVar_size(#tensor, tensor))

namespace nts {
namespace op {

class ntsGraphOp {
public:
  Graph<Empty> *graph_;
  VertexSubset *active_;
    PartitionedGraph *partitioned_graph_;
  ntsGraphOp() {}
  ntsGraphOp(PartitionedGraph *partitioned_graph,VertexSubset *active) {
    graph_ = partitioned_graph->graph_;
    partitioned_graph_=partitioned_graph;
    active_ = active;
  }
  ntsGraphOp(Graph<Empty> *graph) {
    graph_ = graph;
  }
  virtual NtsVar forward(NtsVar &f_input)=0;
  virtual NtsVar forward(NtsVar &f_input, std::vector<VertexId> cacheflag)=0;
  virtual NtsVar backward(NtsVar &output_grad)=0;
};


class ntsNNBaseOp {
public:
  ntsNNBaseOp(){}
  ntsNNBaseOp(int layer_){
  layer=layer_;}
  NtsVar *f_input;
  NtsVar *f_output; 
  int layer=-1;
  virtual NtsVar forward(NtsVar &f_input)=0;
  virtual NtsVar backward(NtsVar &output_grad)=0;
  
};


inline void nts_comp_non_avx256(ValueType *output, ValueType *input, ValueType weight,
          int feat_size) {
  for (int i = 0; i < feat_size; i++) {
    output[i] += input[i] * weight;
  }
}

    inline void shuffle_vec(std::vector<VertexId>& vec) {
        unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count();
        std::shuffle (vec.begin(), vec.end(), std::default_random_engine(seed));
    }
inline void nts_local_shuffle(std::vector<VertexId>& vertex_ids, int batch_size, int super_batch_size ) {
    // 首先是super batch 的shuffle
    // 确定super batch的数量
    VertexId super_batch_num = vertex_ids.size()/super_batch_size;
    // 生成一个数组用于shuffle
    std::vector<VertexId> shuffle_ids(super_batch_num);
    // 填充1到super num的索引
    std::iota(shuffle_ids.begin(), shuffle_ids.end(), 0);
    // 使用C++官方提供的shuffle函数
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffle_ids.begin(), shuffle_ids.end(), g);

    // 将相应的内存部分进行移动
    // 首先复制一份方便进行移动
    std::vector<VertexId> copy_ids(vertex_ids);
    // 获取两个数据指针
    auto* vertex_ids_ptr = vertex_ids.data();
    auto* copy_ids_ptr = copy_ids.data();
    // 根据shuffle id进行复制
    for(VertexId newIndex = 0; newIndex < super_batch_num; newIndex++) {
        auto oldIndex = shuffle_ids[newIndex];
        // 将shuffle位置的数据复制到新位置中
        memcpy(&vertex_ids_ptr[newIndex * super_batch_size], &copy_ids_ptr[oldIndex * super_batch_size],
               sizeof(VertexId) * super_batch_size);
    }

//    // 进行supper batch中batch间的shuffle
//    int batch_num = super_batch_size / batch_size;
//    shuffle_ids.resize(batch_num);
//    std::iota(shuffle_ids.begin(), shuffle_ids.end(), 0);
//    copy_ids = vertex_ids;
//    copy_ids_ptr = copy_ids.data();
//    // 遍历循环，每次去一个super batch进行batch间的shuffle
//    for(VertexId i = 0; i < super_batch_num; i++) {
//        auto* super_batch_data = &vertex_ids_ptr[i * super_batch_size];
//        auto* copy_super_batch_data = &copy_ids_ptr[i * super_batch_size];
//        std::shuffle(shuffle_ids.begin(), shuffle_ids.end(), g);
//        for(VertexId newIndex = 0; newIndex < batch_num; newIndex++) {
//            auto oldIndex = shuffle_ids[newIndex];
//            // 将shuffle位置的数据复制到新位置中
//            memcpy(&super_batch_data[newIndex * batch_size], &copy_super_batch_data[oldIndex * batch_size],
//                   sizeof(VertexId) * batch_size);
//        }
//    }

    // 进行super batch内单个节点的shuffle
    // 重置shuffle_ids
    shuffle_ids.resize(super_batch_size);
    std::iota(shuffle_ids.begin(), shuffle_ids.end(), 0);
    copy_ids = vertex_ids;
    copy_ids_ptr = copy_ids.data();
    // 遍历进行shuffle
    for(VertexId i = 0; i < super_batch_num; i++) {
        auto* super_batch_data = &vertex_ids_ptr[i * super_batch_size];
        auto* copy_batch_data = &copy_ids_ptr[i * super_batch_size];
        // std::iota(shuffle_ids.begin(), shuffle_ids.end(), 0);
        std::shuffle(shuffle_ids.begin(), shuffle_ids.end(), g);
        // for(auto& id : shuffle_ids){
        //     std::cout << id << " " ;
        // }
        // std::cout << std::endl;
        for(VertexId newIndex = 0; newIndex < super_batch_size; newIndex++) {
            auto oldIndex = shuffle_ids[newIndex];
            super_batch_data[newIndex] = copy_batch_data[oldIndex];
            // auto tmp = super_batch_data[oldIndex];
            // super_batch_data[oldIndex] = super_batch_data[newIndex];
            // super_batch_data[newIndex] = tmp;
        }
    }
}

    inline void nts_local_shuffle(std::vector<VertexId>& vertex_ids, int super_batch_size, std::vector<VertexId>& cache_ids,
                                  std::vector<VertexId>& batch_cache_num) {
        // 首先是super batch 的shuffle
        // 确定super batch的数量
        VertexId super_batch_num = vertex_ids.size()/super_batch_size;
        // 生成一个数组用于shuffle
        std::vector<VertexId> shuffle_ids(super_batch_num);
        // 填充1到super num的索引
        std::iota(shuffle_ids.begin(), shuffle_ids.end(), 0);
        // 使用C++官方提供的shuffle函数
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(shuffle_ids.begin(), shuffle_ids.end(), g);

        // 将相应的内存部分进行移动
        // 首先复制一份方便进行移动
        std::vector<VertexId> copy_ids(vertex_ids);
        std::vector<VertexId> copy_cache_ids(cache_ids);
        std::vector<VertexId> copy_batch_cache_num(batch_cache_num);
        // 获取两个数据指针
        auto* vertex_ids_ptr = vertex_ids.data();
        auto* copy_ids_ptr = copy_ids.data();
        // 获取cache id的两个数据指针
        auto* cache_ids_ptr = cache_ids.data();
        auto* copy_cache_ids_ptr = copy_cache_ids.data();

        // 由于cache ids的super batch size不确定，所以先进行累加确定原始节点在每个batch的起始情况
        std::vector<VertexId> cache_super_batch_start(super_batch_num, 0);
        for(VertexId i = 1; i < super_batch_num; i++) {
            if( i >= cache_super_batch_start.size()) {
                std::printf("i: %u, size: %ld, bc size: %ld\n", i, cache_super_batch_start.size(), batch_cache_num.size());
            }
            assert(i < cache_super_batch_start.size());
            cache_super_batch_start[i] = cache_super_batch_start[i-1] + batch_cache_num[i-1];
        }


        // 根据shuffle id进行复制
        VertexId new_cache_offset = 0;
        for(VertexId newIndex = 0; newIndex < super_batch_num; newIndex++) {
            assert(newIndex < shuffle_ids.size());
            auto oldIndex = shuffle_ids[newIndex];
            assert(newIndex * super_batch_size + super_batch_size < vertex_ids.size());
            assert(oldIndex * super_batch_size + super_batch_size < copy_ids.size());
            // 将shuffle位置的数据复制到新位置中
            memcpy(&vertex_ids_ptr[newIndex * super_batch_size], &copy_ids_ptr[oldIndex * super_batch_size],
                   sizeof(VertexId) * super_batch_size);
            assert(new_cache_offset + copy_batch_cache_num[oldIndex] < cache_ids.size());
            assert(cache_super_batch_start[oldIndex] + copy_batch_cache_num[oldIndex] < copy_cache_ids.size());
            assert(newIndex < batch_cache_num.size());
            assert(oldIndex < copy_batch_cache_num.size());
            // 将shuffle后的cache id进行复制
            memcpy(&cache_ids_ptr[new_cache_offset], &copy_cache_ids_ptr[cache_super_batch_start[oldIndex]],
                   sizeof(VertexId) * copy_batch_cache_num[oldIndex]);
            batch_cache_num[newIndex] = copy_batch_cache_num[oldIndex];
            new_cache_offset += copy_batch_cache_num[oldIndex];

        }

//    // 进行supper batch中batch间的shuffle
//    int batch_num = super_batch_size / batch_size;
//    shuffle_ids.resize(batch_num);
//    std::iota(shuffle_ids.begin(), shuffle_ids.end(), 0);
//    copy_ids = vertex_ids;
//    copy_ids_ptr = copy_ids.data();
//    // 遍历循环，每次去一个super batch进行batch间的shuffle
//    for(VertexId i = 0; i < super_batch_num; i++) {
//        auto* super_batch_data = &vertex_ids_ptr[i * super_batch_size];
//        auto* copy_super_batch_data = &copy_ids_ptr[i * super_batch_size];
//        std::shuffle(shuffle_ids.begin(), shuffle_ids.end(), g);
//        for(VertexId newIndex = 0; newIndex < batch_num; newIndex++) {
//            auto oldIndex = shuffle_ids[newIndex];
//            // 将shuffle位置的数据复制到新位置中
//            memcpy(&super_batch_data[newIndex * batch_size], &copy_super_batch_data[oldIndex * batch_size],
//                   sizeof(VertexId) * batch_size);
//        }
//    }

        // 进行super batch内单个节点的shuffle
        // 重置shuffle_ids
        shuffle_ids.resize(super_batch_size);
        std::iota(shuffle_ids.begin(), shuffle_ids.end(), 0);
        copy_ids = vertex_ids;
        copy_ids_ptr = copy_ids.data();
        // 遍历进行shuffle
        for(VertexId i = 0; i < super_batch_num; i++) {
            assert(i*super_batch_size < vertex_ids.size());
            assert(i*super_batch_size < copy_ids.size());
            auto* super_batch_data = &vertex_ids_ptr[i * super_batch_size];
            auto* copy_batch_data = &copy_ids_ptr[i * super_batch_size];
            // std::iota(shuffle_ids.begin(), shuffle_ids.end(), 0);
            std::shuffle(shuffle_ids.begin(), shuffle_ids.end(), g);
            // for(auto& id : shuffle_ids){
            //     std::cout << id << " " ;
            // }
            // std::cout << std::endl;
            for(VertexId newIndex = 0; newIndex < super_batch_size; newIndex++) {
                auto oldIndex = shuffle_ids[newIndex];
                assert(i*super_batch_size + newIndex < vertex_ids.size());
                super_batch_data[newIndex] = copy_batch_data[oldIndex];
                // auto tmp = super_batch_data[oldIndex];
                // super_batch_data[oldIndex] = super_batch_data[newIndex];
                // super_batch_data[newIndex] = tmp;
            }
        }
    }

   inline void print_NtsVar_size(std::string varName, NtsVar& var) {
        int dim = var.dim();
        std::string dim_str = "(";
        for(int i = 0; i < dim; i++) {
            dim_str += std::to_string(var.size(i));
            dim_str += ", ";
        }
        dim_str += ")";
        std::printf("%s size: %s\n", varName.c_str(), dim_str.c_str());
    }


    // TODO: 改进get_most_neighbor
    // 就是先算完所有节点的L阶热点邻居，然后直接取L阶邻居，L阶邻居依赖于L-1阶邻居，会进行递归处理
    // 直接一次性算所有train 节点的L阶邻居
    void get_most_neighbor(std::vector<VertexId>& train_ids, VertexId start, VertexId super_batch_size, VertexId batch_cache_num,
                           VertexId* cache_arr_start, int layers, FullyRepGraph* fully_rep_graph) {
//        whole_graph->global_vertices;
//        whole_graph->column_offset;
//        whole_graph->row_indices;
        std::unordered_map<VertexId, VertexId> sample_count_map;
        // vector具有默认初始0值
        std::vector<VertexId> old_count_vector(fully_rep_graph->global_vertices);
        std::vector<VertexId> new_count_vector(fully_rep_graph->global_vertices);
        // 用batch点初始化old_count_vector
#pragma omp parallel for
        for(VertexId i = start; i < start + super_batch_size ; i++) {
            if(i < train_ids.size()) {
                old_count_vector[train_ids[i]] = 1;
            }
        }

        for(int layer = 1; layer < layers; layer++) {
            // 如果不是第一次的话需要进行清零处理
            if(layer != 1) {
                std::swap(old_count_vector, new_count_vector);
                // 清零new_count_vector重新进行统计
                memset(new_count_vector.data(), 0, sizeof(VertexId) * new_count_vector.size());
            }
            // 使用old_count_vector进行源节点进行迭代，结果输出到new_count_vector中
#pragma omp parallel for
            for(VertexId i = 0; i < old_count_vector.size(); i++) {
                // count不为0的点才是进行迭代的点
                if(old_count_vector[i] > 0) {
                    auto col_start = fully_rep_graph->column_offset[i];
                    auto col_end = fully_rep_graph->column_offset[i+1];
                    for(auto dst_id = col_start; dst_id < col_end; dst_id++) {
                        auto dst = fully_rep_graph->row_indices[dst_id];
                        write_add(&new_count_vector[dst], old_count_vector[i]);
                    }
                }
            }
        }
        // 对得到的vector进行排序
        std::copy(new_count_vector.begin(), new_count_vector.end(),
                  std::back_inserter(old_count_vector));
        std::sort(std::execution::par,old_count_vector.begin(), old_count_vector.end(), [](VertexId a, VertexId b){return a > b;});
        assert(batch_cache_num < old_count_vector.size());
        VertexId pivot = old_count_vector[batch_cache_num];
        // 从new_count_vector里面提取符合的
        auto index = 0u;
#pragma omp parallel for
        for(VertexId i = 0; i < new_count_vector.size(); i++) {
            if(new_count_vector[i] >= pivot) {
                if(index < batch_cache_num) {
                    auto local_index = write_add_return_old(&index, 1u);
                    if(local_index < batch_cache_num){
                        cache_arr_start[local_index] = i;
                    }
                }
            }
        }

//        // TODO: 这里如果不够的话随便用一些节点来填
//        for(VertexId i = 0; i < batch_cache_num && i < pairs.size(); i++) {
//            cache_arr_start[i] = pairs[i].first;
//        }

    }

    int get_most_neighbor(std::vector<VertexId>& train_ids, VertexId start, VertexId super_batch_size, std::vector<VertexId>& batch_cache_num,
                          VertexId super_batch_num, float cache_rate,
                          VertexId* cache_arr_start, int layers, FullyRepGraph* fully_rep_graph) {
//        whole_graph->global_vertices;
//        whole_graph->column_offset;
//        whole_graph->row_indices;
        std::unordered_map<VertexId, VertexId> sample_count_map;
        // vector具有默认初始0值
        std::vector<VertexId> old_count_vector(fully_rep_graph->global_vertices, 0);
        std::vector<VertexId> new_count_vector(fully_rep_graph->global_vertices, 0);
        // 用batch点初始化old_count_vector
#pragma omp parallel for
        for(VertexId i = start; i < start + super_batch_size ; i++) {
            if(i < train_ids.size()) {
                old_count_vector[train_ids[i]] = 1;
            }
        }
        for(int layer = 1; layer < layers; layer++) {
            // 如果不是第一次的话需要进行清零处理
            if(layer != 1) {
                std::swap(old_count_vector, new_count_vector);
                // 清零new_count_vector重新进行统计
                memset(new_count_vector.data(), 0, sizeof(VertexId) * new_count_vector.size());
            }
            // 使用old_count_vector进行源节点进行迭代，结果输出到new_count_vector中
#pragma omp parallel for
            for(VertexId i = 0; i < old_count_vector.size(); i++) {
                // count不为0的点才是进行迭代的点
                if(old_count_vector[i] > 0) {
                    auto col_start = fully_rep_graph->column_offset[i];
                    auto col_end = fully_rep_graph->column_offset[i+1];
                    for(auto dst_id = col_start; dst_id < col_end; dst_id++) {
                        auto dst = fully_rep_graph->row_indices[dst_id];
                        write_add(&new_count_vector[dst], old_count_vector[i]);
                    }
                }
            }
        }

        // 对得到的vector进行排序
        // std::copy(new_count_vector.begin(), new_count_vector.end(),
        //           std::back_inserter(old_count_vector));
        std::memcpy(old_count_vector.data(), new_count_vector.data(), sizeof(VertexId) * new_count_vector.size());
        std::sort(std::execution::par,old_count_vector.begin(), old_count_vector.end(), [](VertexId a, VertexId b){return a > b;});
        VertexId total_sample_num, true_cache_num;
        // #pragma omp parallel for
        for(VertexId i = 0; i < old_count_vector.size(); i++) {
            if(old_count_vector[i] == 0) {
                total_sample_num = i + 1;
                break;
            }
        }
        batch_cache_num[super_batch_num] = total_sample_num * cache_rate;
        true_cache_num = batch_cache_num[super_batch_num];
        VertexId pivot = old_count_vector[true_cache_num];
        // 从new_count_vector里面提取符合的
        auto index = 0u;
#pragma omp parallel for
        for(VertexId i = 0; i < new_count_vector.size(); i++) {
            if(new_count_vector[i] >= pivot) {
                if(index < true_cache_num) {
                    auto local_index = write_add_return_old(&index, 1u);
                    if(local_index < true_cache_num){
                        cache_arr_start[local_index] = i;
                    }
                }
            }
        }
        return true_cache_num;
//        // TODO: 这里如果不够的话随便用一些节点来填
//        for(VertexId i = 0; i < batch_cache_num && i < pairs.size(); i++) {
//            cache_arr_start[i] = pairs[i].first;
//        }

    }


    std::vector<VertexId> preSample(std::vector<VertexId>& train_ids, int batch_size, int batch_cache_num, int layers,
                                    FullyRepGraph* fully_rep_graph, int pipeline_num = 1) {
        int super_batch_num = train_ids.size()/(batch_size*pipeline_num);
        if(super_batch_num * batch_size *pipeline_num < train_ids.size()) {
            super_batch_num++;
        }
        int super_batch_size = batch_size * pipeline_num;
        std::vector<VertexId> batch_cache_ids(batch_cache_num * super_batch_num);
        std::printf("train ids num: %u, cache ids num: %u, super batch num: %u\n", train_ids.size(), batch_cache_ids.size(), super_batch_num);
        // 下面是统计一个superbatch内的节点的数量排名
//        cpu_sampler->sample_fast()
        for(VertexId i = 0; i < super_batch_num; i++) {
            get_most_neighbor(train_ids, super_batch_size * i, super_batch_size, batch_cache_num,
                              &(batch_cache_ids[batch_cache_num * i]), layers, fully_rep_graph);
        }
        return batch_cache_ids;
    }

    std::vector<VertexId> preSample(std::vector<VertexId>& train_ids, int batch_size, std::vector<VertexId>& batch_cache_num,
                                    float cache_rate, VertexId& top_cache_num ,int layers,
                                    FullyRepGraph* fully_rep_graph, float of_rate,
                                    Graph<Empty>* graph, int pipeline_num = 1) {
        
        std::string pre_filename = graph->config->pre_sample_file;
        if(pre_filename.length() < 1 || !file_exists(pre_filename)) {
            size_t last_point = graph->config->edge_file.find_last_of('.');
            pre_filename = graph->config->edge_file.substr(0, last_point+1);
            pre_filename += "pre_sample_b" + std::to_string(graph->config->batch_size) + "_f"
                    + graph->config->fanout_string+ "_p"+ std::to_string(pipeline_num) + ".bin";
            graph->config->cache_rate = 0.8;
            cache_rate = 0.8; //Initialize large hot vertices raito to reduce subsequent pre-sampling overheads
        }

        int super_batch_num = train_ids.size()/(batch_size*pipeline_num);
        if(super_batch_num * batch_size * pipeline_num < train_ids.size()) {
            super_batch_num++;
        }
        int super_batch_size = batch_size * pipeline_num;
        //batch_cache_num负责记录每个super-batch内包含多个点
        batch_cache_num.resize(super_batch_num);
        std::vector<VertexId> batch_cache_ids(cache_rate * fully_rep_graph->global_vertices * super_batch_num);

        std::printf("pre sample filename: %s\n", pre_filename.c_str());
        const char* filename = pre_filename.c_str();
        std::ifstream checkFile(filename);
        if (!checkFile) {
            // 文件不存在，进行采样并写入文件
            // cpu_sampler->sample_fast();

            int cache_num = 0;
            top_cache_num = 0;
            for(VertexId i = 0; i < super_batch_num; i++) {
                int cache_num_i = get_most_neighbor(train_ids, super_batch_size * i, super_batch_size, batch_cache_num, i, cache_rate,
                                                    &batch_cache_ids[cache_num], layers, fully_rep_graph);                           
                cache_num += cache_num_i;
                if(cache_num_i > top_cache_num)
                    top_cache_num = cache_num_i;
            }
            std::printf("after get most\n");
            batch_cache_ids.resize(cache_num);

            // 保存预采样的ID到文件中
            std::ofstream outFile(filename,std::ios::out | std::ios::binary);
            if (!outFile) {
                std::cerr << "Failed to open output file." << std::endl;//标准错误流
                return batch_cache_ids;
            }

            //将每个super-batch包含多少个顶点数的信息写入文件，例如，四个super-batch，就会写入四个数，代表着每个super-batch内的顶点数
            size_t write_count = batch_cache_num.size();
            for(int i = 0; i < write_count; i++){
                std::printf("i:%d batch_cache_num[i]:%d \n", i, batch_cache_num[i]);
            }
            outFile.write(reinterpret_cast<char*>(batch_cache_num.data()), write_count * sizeof(VertexId));
            if (outFile.good()) {
                std::cout << "Data was successfully written to the binary file." << std::endl;
            } else {
                std::cerr << "Error occurred while writing data to the binary file." << std::endl;
            }
            //将每个super-batch内的具体顶点写入文件
            write_count = batch_cache_ids.size();
            outFile.write(reinterpret_cast<char*>(batch_cache_ids.data()), write_count * sizeof(VertexId));
            if (outFile.good()) {
                std::cout << "Data was successfully written to the binary file." << std::endl;
            } else {
                std::cerr << "Error occurred while writing data to the binary file." << std::endl;
            }
            outFile.close();
        }
        else
        {
            // std::ifstream checkFile1(filename);
            std::ifstream checkFile1(filename, std::ios::in | std::ios::binary);

            // 文件存在，从文件中读取预采样的ID
            std::printf("loading data from file to batch_cache_ids\n");
            // checkFile1.open(filename, std::ios::in | std::ios::binary);
            if (!checkFile1.is_open()) {
                fprintf(stderr, "file open failed\n");
                exit(1);
            }
            size_t read_count = batch_cache_num.size();
            std::vector<VertexId> batch_cache_num_tmp(read_count);
            //读出每个super-batch的长度
            checkFile1.read(reinterpret_cast<char*>(batch_cache_num.data()), read_count * sizeof(VertexId));
            int batch_cache_ids_len = 0;
            top_cache_num = 0;
            for(int i = 0; i < read_count; i++){
                if(batch_cache_num[i] * of_rate > top_cache_num)
                    top_cache_num = batch_cache_num[i] * of_rate;
                batch_cache_num_tmp[i] = batch_cache_num[i];
                batch_cache_num[i] = batch_cache_num[i] * of_rate;
                batch_cache_ids_len += batch_cache_num[i]; //取前多少比例的cache node
            }
            batch_cache_ids.resize(batch_cache_ids_len);
            //读入第一个super-batch到batch_cache_ids中，根据给定rate
            batch_cache_ids_len = batch_cache_num_tmp[0] * of_rate;
            checkFile1.read(reinterpret_cast<char*>(batch_cache_ids.data()), (batch_cache_ids_len) * sizeof(VertexId));
            //读入后续super-batch，两个偏移量，文件读入位置偏移，batch_cache_ids写入位置偏移
            VertexId write_pos = batch_cache_ids_len;
            VertexId read_pos = batch_cache_num_tmp[0];
            std::streampos startPosition;
            for(int i = 1; i < read_count; i++){
                batch_cache_ids_len = batch_cache_num[i];
                //移动文件指针到每个super-batch的起始位置
                startPosition = read_pos * sizeof(VertexId) + batch_cache_num_tmp.size() * sizeof(VertexId);
                checkFile1.seekg(startPosition);
                checkFile1.read(reinterpret_cast<char*>(batch_cache_ids.data() + write_pos), (batch_cache_ids_len) * sizeof(VertexId));
                write_pos += batch_cache_ids_len;
                read_pos += batch_cache_num_tmp[i];
            }
            checkFile1.close();
        }
        return batch_cache_ids;
    }

//avx256
inline void nts_comp(ValueType *output, ValueType *input, ValueType weight,
          int feat_size) { 
    const int LEN=8;
  int loop=feat_size/LEN;
  int res=feat_size%LEN;
  __m256 w=_mm256_broadcast_ss(reinterpret_cast<float const *>(&weight));
  for(int i=0;i<loop;i++){
//    __m256 source= *reinterpret_cast<__m256 *>(&(input[i*LEN]));
    __m256 source = _mm256_loadu_ps(&input[i*LEN]);
//      std::printf("6 avx debug 2.2\n");
    __m256 destination= _mm256_loadu_ps(reinterpret_cast<float const *>(&(output[i*LEN])));
    _mm256_storeu_ps(&(output[i*LEN]),_mm256_add_ps(_mm256_mul_ps(source,w),destination));
  }
  for (int i = LEN*loop; i < feat_size; i++) {
    output[i] += input[i] * weight;
  }
}


/**
 * @brief
 * do output += input at feature(array) level
 * @param input input feature
 * @param output output feature
 * @param feat_size feature size
 */
inline void nts_acc(ValueType *output, ValueType *input, int feat_size) {
  for (int i = 0; i < feat_size; i++) {
    // atomic add
    write_add(&output[i], input[i]);
  }
}

inline void nts_acc(ValueType *output, ValueType *input,ValueType weight, int feat_size) {
  for (int i = 0; i < feat_size; i++) {
    // atomic add
    write_add(&output[i], input[i]*weight);
  }
}

/**
 * @brief
 * do output += input at feature(array) level
 * @param input input feature
 * @param output output feature
 * @param feat_size feature size
 */
inline void nts_min(ValueType *output, ValueType *input, VertexId *record, int feat_size, VertexId e_index) {
  for (int i = 0; i < feat_size; i++) {
    // atomic add
    if(write_min(&output[i], input[i])){
        record[i]=e_index;
    }
  }
}

/**
 * @brief
 * do output += input at feature(array) level
 * @param input input feature
 * @param output output feature
 * @param feat_size feature size
 */
inline void nts_max(ValueType *output, ValueType *input, VertexId *record, int feat_size, VertexId e_index) {
  for (int i = 0; i < feat_size; i++) {
    // atomic add
    if(write_max(&output[i], input[i])){
        record[i]=e_index;
    }
  }
}

inline void nts_assign(ValueType *message, ValueType *feature, VertexId* record,
          int feat_size) {
    for(int i=0;i<feat_size;i++){
        message[(record[i]*feat_size)+i]=feature[i];
    }
}
/**
 * @brief
 * copy feature_size elements from b_src[d_offset * feature_size]
 * to d_dst[s_offset * feature_size]
 * @param b_dst dst buffer
 * @param d_offset dst offset, should be a vertex id
 * @param b_src src buffer
 * @param s_offset src offset, should be a vertex id
 * @param feat_size feature size that every vertex have
 */
inline void nts_copy(ValueType *b_dst, long d_offset, ValueType *b_src, VertexId s_offset,
          int feat_size, int counts) {
  // length is the byte level space cost for a vertex feature data
  VertexId length = sizeof(ValueType) * feat_size;
  // LOG_INFO("length %d feat_size %d d_offset %d s_offset
  // %d\n",length,feat_size,d_offset,s_offset);
  memcpy((char *)b_dst + d_offset * length, (char *)b_src + s_offset * length,
         length*counts);
}

/**
 * @brief
 * return 1 / sqrt(out_degree[src] * in_degree[dst]).
 * normally used as edge weight
 * @param src src id
 * @param dst dst id
 * @return ValueType
 */
inline ValueType nts_norm_degree(Graph<Empty> *graph_, VertexId src, VertexId dst) {
//      edge_weight[i] = 1 / (sqrtf(out_degree[source[src]]) * sqrtf(in_degree[destination[warp_id]]));

      return 1 / ((ValueType)std::sqrt(graph_->out_degree_for_backward[src]) *
              (ValueType)std::sqrt(graph_->in_degree_for_backward[dst]));
}

/**
 * @brief
 * get out degree for v
 * @param v vertex id
 * @return ValueType
 */
inline ValueType nts_out_degree(Graph<Empty> *graph_, VertexId v) {
  return (ValueType)(graph_->out_degree_for_backward[v]);
}

/**
 * @brief
 * get in degree for v
 * @param v vertex id
 * @return ValueType
 */
inline ValueType nts_in_degree(Graph<Empty> *graph_, VertexId v) {
  return (ValueType)(graph_->in_degree_for_backward[v]);
}

} // namespace graphop
} // namespace nts



//class ntsNNOp {
//public:
//
//  ntsNNOp() {}
//  virtual NtsVar &forward(NtsVar &input) = 0;
//  virtual NtsVar backward(NtsVar &output_grad) = 0;
//};

#endif
