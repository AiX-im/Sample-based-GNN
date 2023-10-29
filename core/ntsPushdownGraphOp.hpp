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
#ifndef NTSPUSHDOWNGRAPHOP_HPP
#define NTSPUSHDOWNGRAPHOP_HPP
#include <assert.h>
#include <map>
#include <math.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <atomic>

#include "core/graph.hpp"
#include "core/ntsBaseOp.hpp"
#include "core/ntsPeerRPC.hpp"
#include "core/ntsFastSampler.hpp"
namespace nts {
namespace op {
class PushDownOp : public ntsGraphOp{
public:
  SampledSubgraph* subgraphs;
  int layer=0;
  ValueType *f_embedding_output;
  PushDownOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_)
      : ntsGraphOp(graph_) {
    subgraphs = subgraphs_;
    layer=layer_;
  }

  PushDownOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,ValueType *f_embedding_output_,int layer_)
      : ntsGraphOp(graph_) {
    subgraphs = subgraphs_;
    layer=layer_; 
    f_embedding_output = f_embedding_output_;   
  }

  NtsVar forward(NtsVar &f_input){
    int feature_size = f_input.size(1);
//    std::printf("forward size (%lu, %lu)\n", subgraphs->sampled_sgs[layer]->dst().size(), feature_size);
    NtsVar f_output=graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->dst().size(), 
                feature_size},torch::DeviceType::CPU);     
//    std::printf("debug 1\n");
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer = 
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
    // memset(f_embedding_output,0,(long)(subgraphs->sampled_sgs[layer]->dst().size())*feature_size*sizeof(ValueType));
//    std::printf("debug 2\n");
    this->subgraphs->compute_one_layer(
            [&](VertexId local_dst, std::vector<VertexId>& column_offset, 
                std::vector<VertexId>& row_indices){
                VertexId src_start=column_offset[local_dst];
                VertexId src_end=column_offset[local_dst+1];
                VertexId dst=subgraphs->sampled_sgs[layer]->dst()[local_dst];

                ValueType *local_output=f_output_buffer+local_dst*feature_size;
                
                for(VertexId src_offset=src_start;
                        src_offset<src_end;src_offset++){
                    VertexId local_src=row_indices[src_offset];
                    VertexId src=subgraphs->sampled_sgs[layer]->src()[local_src];
                    ValueType *local_input = f_input_buffer + src * feature_size;
                    nts_comp(local_output, local_input,
                            nts_norm_degree(graph_,src, dst), feature_size);
                      //  nts_comp(local_output, local_input,
                      //        weight[src_offset], feature_size);
                }
                // TODO: 这里可能有bug
                 memcpy(f_embedding_output+local_dst*feature_size,
                     local_output,feature_size*sizeof(ValueType));
              },
            layer
            );   
      return f_output;
   }
   
  NtsVar forward(NtsVar &f_input, std::vector<VertexId> cacheflag){
    int feature_size = f_input.size(1);
    NtsVar f_output=graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->dst().size(), 
                feature_size},torch::DeviceType::CPU);
      return f_output;
   }
   NtsVar backward(NtsVar &f_output_grad){
        int feature_size=f_output_grad.size(1);
        //std::printf("output size: (%lu, %lu)\n", f_output_grad.size(0), f_output_grad.size(1));
        NtsVar f_input_grad=graph_->Nts->NewLeafTensor({subgraphs->sampled_sgs[layer]->src().size(), 
                feature_size},torch::DeviceType::CPU);
       return f_input_grad;
   }

};

    class PushDownBatchOp : public ntsGraphOp{
        SampledSubgraph* subgraphs;
        int layer=0;
        int batch_start;
        int batch_end;

    public:
        NtsVar forward(NtsVar& f_input) {
//            std::printf("batch start: %d, batch end: %d\n", batch_start, batch_end);
            int feature_size = f_input.size(1);
//    std::printf("forward size (%lu, %lu)\n", subgraphs->sampled_sgs[layer]->dst().size(), feature_size);
            NtsVar f_output=graph_->Nts->NewKeyTensor({batch_end - batch_start, feature_size},
                                                      torch::DeviceType::CPU);
            // CPU的不需要梯度，故这里设置为不需梯度
            f_output.requires_grad_(false);
//    std::printf("debug 1\n");
            ValueType *f_input_buffer =
                    graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
            ValueType *f_output_buffer =
                    graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
            auto& forward_weight =subgraphs->sampled_sgs[layer]->e_w_f();
            this->subgraphs->compute_one_layer_batch(
                    [&](VertexId local_dst, std::vector<VertexId>& column_offset,
                        std::vector<VertexId>& row_indices){
                        VertexId src_start=column_offset[local_dst];
                        VertexId src_end=column_offset[local_dst+1];
                        VertexId dst=subgraphs->sampled_sgs[layer]->dst()[local_dst];

                        // Toao修改为相对于batch的偏移量
                        assert(local_dst >= batch_start && local_dst < batch_end);

                        ValueType *local_output=f_output_buffer+(local_dst - batch_start)*feature_size;

                        for(VertexId src_offset=src_start;
                            src_offset<src_end;src_offset++){
                            VertexId local_src=row_indices[src_offset];
                            // 根据全局的src进行调整，所以传进来的是所有的feature
                            VertexId src=subgraphs->sampled_sgs[layer]->src()[local_src];
                            ValueType *local_input = f_input_buffer + src * feature_size;
                            nts_comp(local_output, local_input,
                                     forward_weight[src_offset], feature_size);
//                            nts_comp(local_output, local_input,
//                                     nts_norm_degree(graph_,src, dst), feature_size);
//                            assert(subgraphs->sampled_sgs[layer]->e_w_f()[src_offset] - nts_norm_degree(graph_, src, dst) < 1e-4);
//                            auto weight = nts_norm_degree(graph_,src, dst);
//                            ++weight_num;
//                            auto old_sum = weight_sum.load();
//                            while(!weight_sum.compare_exchange_weak(old_sum, old_sum+weight)){
//                                old_sum = weight_sum.load();
//                            }
//                            for(int i = 0; i < feature_size; i++){
//                                local_output[i] += local_input[i]*weight;
//                            }
                            //  nts_comp(local_output, local_input,
                            //        weight[src_offset], feature_size);
                        }
                    },
                    layer, batch_start, batch_end
            );
            return f_output;
        }
        NtsVar forward(NtsVar &f_input, std::vector<VertexId> cacheflag){

            int feature_size = f_input.size(1);
//    std::printf("forward size (%lu, %lu)\n", subgraphs->sampled_sgs[layer]->dst().size(), feature_size);
            NtsVar f_output=graph_->Nts->NewKeyTensor({batch_end - batch_start, feature_size},
                                                      torch::DeviceType::CPU);
            return f_output;
        }
        NtsVar backward(NtsVar &f_output_grad){
            int feature_size=f_output_grad.size(1);
            //std::printf("output size: (%lu, %lu)\n", f_output_grad.size(0), f_output_grad.size(1));
            NtsVar f_input_grad=graph_->Nts->NewLeafTensor({subgraphs->sampled_sgs[layer]->src().size(),
                                                            feature_size},torch::DeviceType::CPU);
            return f_input_grad;
        }

        PushDownBatchOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_, int batch_start,
                        int batch_end)
                : ntsGraphOp(graph_) {
            subgraphs = subgraphs_;
            layer=layer_;
            this->batch_start = batch_start;
            this->batch_end = batch_end;

        }
    };


    class CPUPushDownOp : public ntsGraphOp{
        SampledSubgraph* subgraphs;
        int layer=0;

    public:
        NtsVar forward(NtsVar& f_input) {
            int feature_size = f_input.size(1);
//    std::printf("forward size (%lu, %lu)\n", subgraphs->sampled_sgs[layer]->dst().size(), feature_size);
            NtsVar f_output=graph_->Nts->NewKeyTensor({static_cast<long>(subgraphs->sampled_sgs[layer]->dst().size()), feature_size},
                                                      torch::DeviceType::CPU);
//            std::printf("input size: %d, output size: %d\n", f_input.size(0), f_output.size(0));
            // CPU的不需要梯度，故这里设置为不需梯度
            f_output.requires_grad_(false);
//    std::printf("debug 1\n");
            ValueType *f_input_buffer =
                    graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
            ValueType *f_output_buffer =
                    graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
            auto& forward_weight =subgraphs->sampled_sgs[layer]->e_w_f();
            this->subgraphs->compute_one_layer(
                    [&](VertexId local_dst, std::vector<VertexId>& column_offset,
                        std::vector<VertexId>& row_indices){
                        VertexId src_start=column_offset[local_dst];
                        VertexId src_end=column_offset[local_dst+1];
                        VertexId dst=subgraphs->sampled_sgs[layer]->dst()[local_dst];

                        // Toao修改为相对于batch的偏移量

                        ValueType *local_output=f_output_buffer+(local_dst)*feature_size;

                        for(VertexId src_offset=src_start;
                            src_offset<src_end;src_offset++){
                            VertexId local_src=row_indices[src_offset];
                            // 根据全局的src进行调整，所以传进来的是所有的feature
                            VertexId src=subgraphs->sampled_sgs[layer]->src()[local_src];
                            ValueType *local_input = f_input_buffer + src * feature_size;
                            nts_comp(local_output, local_input,
                                     forward_weight[src_offset], feature_size);
//                            nts_comp(local_output, local_input,
//                                     nts_norm_degree(graph_,src, dst), feature_size);
//                            assert(subgraphs->sampled_sgs[layer]->e_w_f()[src_offset] - nts_norm_degree(graph_, src, dst) < 1e-4);
//                            auto weight = nts_norm_degree(graph_,src, dst);
//                            ++weight_num;
//                            auto old_sum = weight_sum.load();
//                            while(!weight_sum.compare_exchange_weak(old_sum, old_sum+weight)){
//                                old_sum = weight_sum.load();
//                            }
//                            for(int i = 0; i < feature_size; i++){
//                                local_output[i] += local_input[i]*weight;
//                            }
                            //  nts_comp(local_output, local_input,
                            //        weight[src_offset], feature_size);
                        }
                    },
                    layer
            );
            return f_output;
        }
        NtsVar forward(NtsVar &f_input, std::vector<VertexId> cacheflag){

            int feature_size = f_input.size(1);
//    std::printf("forward size (%lu, %lu)\n", subgraphs->sampled_sgs[layer]->dst().size(), feature_size);
            NtsVar f_output=graph_->Nts->NewKeyTensor({f_input.size(0), feature_size},
                                                      torch::DeviceType::CPU);
            return f_output;
        }
        NtsVar backward(NtsVar &f_output_grad){
            int feature_size=f_output_grad.size(1);
            //std::printf("output size: (%lu, %lu)\n", f_output_grad.size(0), f_output_grad.size(1));
            NtsVar f_input_grad=graph_->Nts->NewLeafTensor({subgraphs->sampled_sgs[layer]->src().size(),
                                                            feature_size},torch::DeviceType::CPU);
            return f_input_grad;
        }

        CPUPushDownOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_)
                : ntsGraphOp(graph_) {
            subgraphs = subgraphs_;
            layer=layer_;

        }
    };

    class GPUPushDownBatchOp : public ntsGraphOp{
        SampledSubgraph* subgraphs;
        int layer=0;
        int batch_start;
        int batch_end;
        Cuda_Stream* cuda_stream;

    public:
        NtsVar forward(NtsVar& f_input) {

            int feature_size = f_input.size(1);
//    std::printf("forward size (%lu, %lu)\n", subgraphs->sampled_sgs[layer]->dst().size(), feature_size);
            // TODO: 这里要判断一下是不是所有点
            assert(batch_end - batch_start == subgraphs->sampled_sgs[layer]->v_size);
            NtsVar f_output=graph_->Nts->NewKeyTensor({batch_end - batch_start, feature_size},
                                                      torch::DeviceType::CUDA);
            // CPU的不需要梯度，故这里设置为不需梯度
            f_output.requires_grad_(false);
//    std::printf("debug 1\n");
            ValueType *f_input_buffer =
                    graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CUDA);
            ValueType *f_output_buffer =
                    graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CUDA);

            float* weight_forward=subgraphs->sampled_sgs[layer]->dev_e_w();
            VertexId* row_indices=subgraphs->sampled_sgs[layer]->dev_r_i();
            VertexId* column_offset=subgraphs->sampled_sgs[layer]->dev_c_o();
            VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
            VertexId batch_size=subgraphs->sampled_sgs[layer]->v_size;

            int column_num = subgraphs->sampled_sgs[layer]->src_size;
            cuda_stream->Gather_By_Dst_From_Src_Spmm(
                    f_input_buffer, f_output_buffer, weight_forward, // data
                    row_indices, column_offset, column_num,0, 0, 0, 0,
                    edge_size, batch_size,feature_size, true, false);

            return f_output;
        }
        NtsVar forward(NtsVar &f_input, std::vector<VertexId> cacheflag){

            int feature_size = f_input.size(1);
//    std::printf("forward size (%lu, %lu)\n", subgraphs->sampled_sgs[layer]->dst().size(), feature_size);
            NtsVar f_output=graph_->Nts->NewKeyTensor({batch_end - batch_start, feature_size},
                                                      torch::DeviceType::CPU);
            return f_output;
        }
        NtsVar backward(NtsVar &f_output_grad){
            int feature_size=f_output_grad.size(1);
            //std::printf("output size: (%lu, %lu)\n", f_output_grad.size(0), f_output_grad.size(1));
            NtsVar f_input_grad=graph_->Nts->NewLeafTensor({subgraphs->sampled_sgs[layer]->src().size(),
                                                            feature_size},torch::DeviceType::CPU);
            return f_input_grad;
        }

        GPUPushDownBatchOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_, int batch_start,
                        int batch_end, Cuda_Stream* cudaStream)
                : ntsGraphOp(graph_) {
            subgraphs = subgraphs_;
            layer=layer_;
            this->batch_start = batch_start;
            this->batch_end = batch_end;
            this->cuda_stream = cudaStream;
        }
    };

    class BatchGPUScatterSrc : public ntsGraphOp {
    private:
        Cuda_Stream* cudaStream;
        SampledSubgraph* subgraphs;
        int layer=0;
    public:
        BatchGPUScatterSrc(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_, Cuda_Stream* cudaStream)
                : ntsGraphOp(graph_) {
            this->subgraphs = subgraphs_;
            this->layer = layer_;
            this->cudaStream = cudaStream;
        }
        NtsVar forward(NtsVar &f_input){
            int feature_size = f_input.size(1);
            // printf("forward feature:%d layer:%d\n",feature_size,layer);
            // NtsVar f_output = subgraphs->forward_embedding[layer];
            // graph_->Nts->ZeroVarMem(f_output);
            assert(subgraphs->sampled_sgs[layer]->e_size != 0);
            NtsVar f_output = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->e_size,feature_size},torch::DeviceType::CUDA);

            ValueType *f_input_buffer =
                    graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CUDA);
            ValueType *f_output_buffer =
                    graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CUDA);
            float* weight_forward=subgraphs->sampled_sgs[layer]->dev_e_w();
            VertexId* row_indices=subgraphs->sampled_sgs[layer]->dev_r_i();
            VertexId* column_offset=subgraphs->sampled_sgs[layer]->dev_c_o();
            VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
            VertexId batch_size=subgraphs->sampled_sgs[layer]->v_size;

            int column_num = subgraphs->sampled_sgs[layer]->src_size;
//            std::printf("BatchGPUScatterSrc layer: %d, batch size: %d, column num: %d\n", layer, batch_size, column_num);
            // TODO: 这个类的前向和反向都需要添加点的映射转换
            cudaStream->Scatter_Src_to_Msg(f_output_buffer, f_input_buffer, row_indices,
                                           column_offset, batch_size,  feature_size);

            return f_output;
        }
        NtsVar forward(NtsVar &f_input,std::vector<VertexId> cacheflag){
            NtsVar f_output = graph_->Nts->NewKeyTensor(f_input, torch::DeviceType::CPU);
            return f_output;
        }
        NtsVar backward(NtsVar &f_output_grad){
            int feature_size = f_output_grad.size(1);
            // printf("backward feature:%d layer:%d\n",feature_size,layer);
            // NtsVar f_input_grad = graph_->Nts->NewKeyTensor({subgraphs->layer_size[0],feature_size},torch::DeviceType::CUDA);
            //graph_->Nts->ZeroVarMem(f_input_grad);
            // TODO: 将src_size改为了local size
            NtsVar f_input_grad = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->src_size,feature_size},torch::DeviceType::CUDA);
            ValueType *f_input_grad_buffer =
                    graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CUDA);
            ValueType *f_output_grad_buffer =
                    graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CUDA);
            VertexId* row_indices=subgraphs->sampled_sgs[layer]->dev_r_i();
            VertexId* column_offset=subgraphs->sampled_sgs[layer]->dev_c_o();
            ValueType *weight_backward = subgraphs->sampled_sgs[layer]->dev_e_w();
            VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
            VertexId batch_size=subgraphs->sampled_sgs[layer]->v_size;
            int column_num = subgraphs->sampled_sgs[layer]->src_size;

            cudaStream->Gather_Msg_To_Src(f_input_grad_buffer, f_output_grad_buffer,
                                          row_indices, column_offset, batch_size, feature_size);

//    std::printf("param grad size: (%d, %d)\n", f_output_grad.size(0), f_output_grad.size(1));
//     std::printf("return grad size: (%d, %d)\n", f_input_grad.size(0), f_input_grad.size(1));
//     std::printf("batch size: %d, column num: %d\n", batch_size, f_input_grad.size(0));
            return f_input_grad;
        }


    };


    class BatchGPUScatterDst : public ntsGraphOp {
    private:
        Cuda_Stream* cudaStream;
        SampledSubgraph* subgraphs;
        int layer=0;
    public:
        BatchGPUScatterDst(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_, Cuda_Stream* cudaStream)
                : ntsGraphOp(graph_) {
            this->subgraphs = subgraphs_;
            this->layer = layer_;
            this->cudaStream = cudaStream;
        }
        NtsVar forward(NtsVar &f_input){
            int feature_size = f_input.size(1);
            // printf("forward feature:%d layer:%d\n",feature_size,layer);
            // NtsVar f_output = subgraphs->forward_embedding[layer];
            // graph_->Nts->ZeroVarMem(f_output);
            assert(subgraphs->sampled_sgs[layer]->e_size != 0);
            NtsVar f_output = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->e_size,feature_size},torch::DeviceType::CUDA);

            ValueType *f_input_buffer =
                    graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CUDA);
            ValueType *f_output_buffer =
                    graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CUDA);
            float* weight_forward=subgraphs->sampled_sgs[layer]->dev_e_w();
            VertexId* row_indices=subgraphs->sampled_sgs[layer]->dev_r_i();
            VertexId* column_offset=subgraphs->sampled_sgs[layer]->dev_c_o();
            VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
            VertexId batch_size=subgraphs->sampled_sgs[layer]->v_size;

            int column_num = subgraphs->sampled_sgs[layer]->src_size;
//            std::printf("BatchGPUScatterDst layer: %d batch size: %d, column num: %d\n", layer, batch_size, column_num);

            cudaStream->Scatter_Dst_to_Msg_Map(f_output_buffer, f_input_buffer, row_indices,
                                           column_offset, batch_size,  feature_size, subgraphs->sampled_sgs[layer]->dev_dst_local_id);

            return f_output;
        }
        NtsVar forward(NtsVar &f_input,std::vector<VertexId> cacheflag){
            NtsVar f_output = graph_->Nts->NewKeyTensor(f_input, torch::DeviceType::CPU);
            return f_output;
        }
        NtsVar backward(NtsVar &f_output_grad){
            int feature_size = f_output_grad.size(1);
            // printf("backward feature:%d layer:%d\n",feature_size,layer);
            // NtsVar f_input_grad = graph_->Nts->NewKeyTensor({subgraphs->layer_size[0],feature_size},torch::DeviceType::CUDA);
            //graph_->Nts->ZeroVarMem(f_input_grad);
            // TODO: 下面的v_size改为src size
            NtsVar f_input_grad = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->src_size,feature_size},torch::DeviceType::CUDA);
            ValueType *f_input_grad_buffer =
                    graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CUDA);
            ValueType *f_output_grad_buffer =
                    graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CUDA);
            VertexId* row_indices=subgraphs->sampled_sgs[layer]->dev_r_i();
            VertexId* column_offset=subgraphs->sampled_sgs[layer]->dev_c_o();
            ValueType *weight_backward = subgraphs->sampled_sgs[layer]->dev_e_w();
            VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
            VertexId batch_size=subgraphs->sampled_sgs[layer]->v_size;
            int column_num = subgraphs->sampled_sgs[layer]->src_size;

            cudaStream->Gather_Msg_to_Dst_Map(f_input_grad_buffer, f_output_grad_buffer,
                                          row_indices, column_offset, batch_size, feature_size,
                                              subgraphs->sampled_sgs[layer]->dev_dst_local_id);

//    std::printf("param grad size: (%d, %d)\n", f_output_grad.size(0), f_output_grad.size(1));
//     std::printf("return grad size: (%d, %d)\n", f_input_grad.size(0), f_input_grad.size(1));
//     std::printf("batch size: %d, column num: %d\n", batch_size, f_input_grad.size(0));
            return f_input_grad;
        }


    };

    class BatchGPUSrcDstScatterOp : public ntsGraphOp {
    private:
        Cuda_Stream* cudaStream;
        SampledSubgraph* subgraphs;
        int layer=0;
        int device_id;
    public:
        BatchGPUSrcDstScatterOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_,
                                Cuda_Stream* cudaStream, int device_id_=0)
                : ntsGraphOp(graph_), device_id(device_id_) {
            this->subgraphs = subgraphs_;
            this->layer = layer_;
            this->cudaStream = cudaStream;
        }
        NtsVar forward(NtsVar &f_input){
            int feature_size = f_input.size(1);
            // printf("forward feature:%d layer:%d\n",feature_size,layer);
            // NtsVar f_output = subgraphs->forward_embedding[layer];
            // graph_->Nts->ZeroVarMem(f_output);
            assert(subgraphs->sampled_sgs[layer]->e_size != 0);
            NtsVar f_output = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->e_size,feature_size * 2},
                                                        torch::DeviceType::CUDA, device_id);

            ValueType *f_input_buffer =
                    graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CUDA);
            ValueType *f_output_buffer =
                    graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CUDA);
            float* weight_forward=subgraphs->sampled_sgs[layer]->dev_e_w();
            VertexId* row_indices=subgraphs->sampled_sgs[layer]->dev_r_i();
            VertexId* column_offset=subgraphs->sampled_sgs[layer]->dev_c_o();
            VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
            VertexId batch_size=subgraphs->sampled_sgs[layer]->v_size;

            int column_num = subgraphs->sampled_sgs[layer]->src_size;
//            std::printf("BatchGPUScatterSrc layer: %d, batch size: %d, column num: %d\n", layer, batch_size, column_num);
            // TODO: 这个类的前向和反向都需要添加点的映射转换
            cudaStream->Scatter_Src_Dst_to_Msg(f_output_buffer, f_input_buffer, row_indices,
                                           column_offset, batch_size,  feature_size, subgraphs->sampled_sgs[layer]->dev_dst_local_id);

            return f_output;
        }
        NtsVar forward(NtsVar &f_input,std::vector<VertexId> cacheflag){
            NtsVar f_output = graph_->Nts->NewKeyTensor(f_input, torch::DeviceType::CPU);
            return f_output;
        }
        NtsVar backward(NtsVar &f_output_grad){
            int feature_size = f_output_grad.size(1);
            // printf("backward feature:%d layer:%d\n",feature_size,layer);
            // NtsVar f_input_grad = graph_->Nts->NewKeyTensor({subgraphs->layer_size[0],feature_size},torch::DeviceType::CUDA);
            //graph_->Nts->ZeroVarMem(f_input_grad);
            // TODO: 将src_size改为了local size
            NtsVar f_input_grad = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->src_size,feature_size/2},
                                                            torch::DeviceType::CUDA, device_id);
            ValueType *f_input_grad_buffer =
                    graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CUDA);
            ValueType *f_output_grad_buffer =
                    graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CUDA);
            VertexId* row_indices=subgraphs->sampled_sgs[layer]->dev_r_i();
            VertexId* column_offset=subgraphs->sampled_sgs[layer]->dev_c_o();
            ValueType *weight_backward = subgraphs->sampled_sgs[layer]->dev_e_w();
            VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
            VertexId batch_size=subgraphs->sampled_sgs[layer]->v_size;
            int column_num = subgraphs->sampled_sgs[layer]->src_size;
            feature_size /= 2;

//            print_tensor_size(f_output_grad);
//            print_tensor_size(f_input_grad);
//            std::printf("dst size: %d, src size: %d, edge size: %d\n", subgraphs->sampled_sgs[layer]->v_size,
//                        subgraphs->sampled_sgs[layer]->src_size, subgraphs->sampled_sgs[layer]->e_size);
            cudaStream->Gather_Msg_To_Src_Dst(f_input_grad_buffer, f_output_grad_buffer,
                                          row_indices, column_offset, batch_size, feature_size,
                                          subgraphs->sampled_sgs[layer]->dev_dst_local_id);

//    std::printf("param grad size: (%d, %d)\n", f_output_grad.size(0), f_output_grad.size(1));
//     std::printf("return grad size: (%d, %d)\n", f_input_grad.size(0), f_input_grad.size(1));
//     std::printf("batch size: %d, column num: %d\n", batch_size, f_input_grad.size(0));
//            int before_max_index =  f_output_grad.abs().sum(1).argmax().item<int>();
//            int after_max_index = f_input_grad.abs().sum(1).argmax().item<int>();
//            std::printf("input agrad max id: %d, sum: %.4lf\n", before_max_index, f_output_grad[before_max_index].abs().sum().item<double>());
//            std::printf("BatchGPUSrcDstScatterOp input grad sum: %.4lf\n", f_output_grad.abs().sum().item<double>());
//            std::printf("output agrad max id: %d, sum: %.4lf\n", after_max_index, f_input_grad[after_max_index].abs().sum().item<double>());
//            std::printf("BatchGPUSrcDstScatterOp output grad sum: %.4lf\n", f_input_grad.abs().sum().item<double>());
            return f_input_grad;
        }


    };

    class BatchGPUEdgeSoftMax : public ntsGraphOp {
    private:
        Cuda_Stream* cudaStream;
        SampledSubgraph* subgraphs;
        int layer=0;
        NtsVar IntermediateResult;
        int device_id;
    public:
        BatchGPUEdgeSoftMax(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_,
                            Cuda_Stream* cudaStream, int device_id_=0)
                : ntsGraphOp(graph_), device_id(device_id_) {
            this->subgraphs = subgraphs_;
            this->layer = layer_;
            this->cudaStream = cudaStream;
        }
        NtsVar forward(NtsVar &f_input){
            int feature_size = f_input.size(1);
            // printf("forward feature:%d layer:%d\n",feature_size,layer);
            // NtsVar f_output = subgraphs->forward_embedding[layer];
            // graph_->Nts->ZeroVarMem(f_output);
            assert(subgraphs->sampled_sgs[layer]->e_size != 0);
            NtsVar f_output = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->e_size,feature_size},
                                                        torch::DeviceType::CUDA, device_id);
            IntermediateResult = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->e_size,feature_size},
                                                           torch::DeviceType::CUDA, device_id);

            ValueType *f_input_buffer =
                    graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CUDA);
            ValueType *f_output_buffer =
                    graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CUDA);
            ValueType *f_cache_buffer =
                    graph_->Nts->getWritableBuffer(IntermediateResult, torch::DeviceType::CUDA);
            float* weight_forward=subgraphs->sampled_sgs[layer]->dev_e_w();
            VertexId* row_indices=subgraphs->sampled_sgs[layer]->dev_r_i();
            VertexId* column_offset=subgraphs->sampled_sgs[layer]->dev_c_o();
            VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
            VertexId batch_size=subgraphs->sampled_sgs[layer]->v_size;

            int column_num = subgraphs->sampled_sgs[layer]->src_size;
//            std::printf("BatchGPUEdgeSoftMax layer: %d batch size: %d, column num: %d\n", layer, batch_size, column_num);

//            cudaStream->Edge_Softmax_Forward_Block(f_output_buffer,f_input_buffer,
//                                                    f_cache_buffer,
//                                                    row_indices,column_offset,
//                                                   batch_size,feature_size);

            cudaStream->Edge_Softmax_Forward_Norm_Block(f_output_buffer,f_input_buffer,
                                                   f_cache_buffer,
                                                   row_indices,column_offset,
                                                   batch_size,feature_size);
            return f_output;
        }
        NtsVar forward(NtsVar &f_input,std::vector<VertexId> cacheflag){
            NtsVar f_output = graph_->Nts->NewKeyTensor(f_input, torch::DeviceType::CPU);
            return f_output;
        }
        NtsVar backward(NtsVar &f_output_grad){
            int feature_size = f_output_grad.size(1);
            // printf("backward feature:%d layer:%d\n",feature_size,layer);
            // NtsVar f_input_grad = graph_->Nts->NewKeyTensor({subgraphs->layer_size[0],feature_size},torch::DeviceType::CUDA);
            //graph_->Nts->ZeroVarMem(f_input_grad);
            NtsVar f_input_grad = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->e_size,feature_size},
                                                            torch::DeviceType::CUDA, device_id);
            ValueType *f_input_grad_buffer =
                    graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CUDA);
            ValueType *f_output_grad_buffer =
                    graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CUDA);
            ValueType *f_cache_buffer =
                    graph_->Nts->getWritableBuffer(IntermediateResult, torch::DeviceType::CUDA);
            VertexId* row_indices=subgraphs->sampled_sgs[layer]->dev_r_i();
            VertexId* column_offset=subgraphs->sampled_sgs[layer]->dev_c_o();
            ValueType *weight_backward = subgraphs->sampled_sgs[layer]->dev_e_w();
            VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
            VertexId batch_size=subgraphs->sampled_sgs[layer]->v_size;
            int column_num = subgraphs->sampled_sgs[layer]->src_size;

            cudaStream->Edge_Softmax_Backward_Block(f_input_grad_buffer,f_output_grad_buffer,
                                                     f_cache_buffer,
                                                     row_indices,column_offset,
                                                    batch_size,feature_size);

//    std::printf("param grad size: (%d, %d)\n", f_output_grad.size(0), f_output_grad.size(1));
//     std::printf("return grad size: (%d, %d)\n", f_input_grad.size(0), f_input_grad.size(1));
//     std::printf("batch size: %d, column num: %d\n", batch_size, f_input_grad.size(0));

//            std::printf("BatchGPUEdgeSoftMax input grad sum: %.4lf\n", f_output_grad.abs().sum().item<double>());
//            std::printf("BatchGPUEdgeSoftMax output grad sum: %.4lf\n", f_input_grad.abs().sum().item<double>());
            return f_input_grad;
        }
    };


    class BatchGPUAggregateDst : public ntsGraphOp {
    private:
        Cuda_Stream* cudaStream;
        SampledSubgraph* subgraphs;
        int layer=0;
        int device_id;
    public:
        BatchGPUAggregateDst(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_,
                             Cuda_Stream* cudaStream, int device_id_=0)
                : ntsGraphOp(graph_), device_id(device_id_) {
            this->subgraphs = subgraphs_;
            this->layer = layer_;
            this->cudaStream = cudaStream;
        }
        NtsVar forward(NtsVar &f_input){
            int feature_size = f_input.size(1);
            // printf("forward feature:%d layer:%d\n",feature_size,layer);
            // NtsVar f_output = subgraphs->forward_embedding[layer];
            // graph_->Nts->ZeroVarMem(f_output);
            assert(subgraphs->sampled_sgs[layer]->e_size != 0);
            NtsVar f_output = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->v_size,feature_size},
                                                        torch::DeviceType::CUDA, device_id);
//            LOG_INFO("v_size: %d", subgraphs->sampled_sgs[layer]->v_size);
            ValueType *f_input_buffer =
                    graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CUDA);
            ValueType *f_output_buffer =
                    graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CUDA);
            float* weight_forward=subgraphs->sampled_sgs[layer]->dev_e_w();
            VertexId* row_indices=subgraphs->sampled_sgs[layer]->dev_r_i();
            VertexId* column_offset=subgraphs->sampled_sgs[layer]->dev_c_o();
            VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
            VertexId batch_size=subgraphs->sampled_sgs[layer]->v_size;

            int column_num = subgraphs->sampled_sgs[layer]->src_size;
//            std::printf("BatchGPUAggregateDst layer: %d batch size: %d, column num: %d\n", layer, batch_size, column_num);
            // TODO: 这里gather的话也应该是gather到local size大小，不过最后一层只会gather到batch size大小，即要分开两个gather，
            //  一个gather到local size，一个gather到batch size
            cudaStream->Gather_Msg_to_Dst(f_output_buffer,f_input_buffer,
                                           row_indices,column_offset,
                                          batch_size,feature_size);

            return f_output;
        }
        NtsVar forward(NtsVar &f_input,std::vector<VertexId> cacheflag){
            NtsVar f_output = graph_->Nts->NewKeyTensor(f_input, torch::DeviceType::CPU);
            return f_output;
        }
        NtsVar backward(NtsVar &f_output_grad){
            int feature_size = f_output_grad.size(1);
            // printf("backward feature:%d layer:%d\n",feature_size,layer);
            // NtsVar f_input_grad = graph_->Nts->NewKeyTensor({subgraphs->layer_size[0],feature_size},torch::DeviceType::CUDA);
            //graph_->Nts->ZeroVarMem(f_input_grad);
            NtsVar f_input_grad = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->e_size,feature_size},
                                                            torch::DeviceType::CUDA, device_id);
            ValueType *f_input_grad_buffer =
                    graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CUDA);
            ValueType *f_output_grad_buffer =
                    graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CUDA);
            VertexId* row_indices=subgraphs->sampled_sgs[layer]->dev_r_i();
            VertexId* column_offset=subgraphs->sampled_sgs[layer]->dev_c_o();
            ValueType *weight_backward = subgraphs->sampled_sgs[layer]->dev_e_w();
            VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
            VertexId batch_size=subgraphs->sampled_sgs[layer]->v_size;
            int column_num = subgraphs->sampled_sgs[layer]->src_size;

            cudaStream->Scatter_Dst_to_Msg(f_input_grad_buffer,f_output_grad_buffer,
                                            row_indices, column_offset,
                                           batch_size, feature_size);
//    std::printf("param grad size: (%d, %d)\n", f_output_grad.size(0), f_output_grad.size(1));
//     std::printf("return grad size: (%d, %d)\n", f_input_grad.size(0), f_input_grad.size(1));
//     std::printf("batch size: %d, column num: %d\n", batch_size, f_input_grad.size(0));
//            std::printf("BatchGPUAggregateDst input grad sum: %.4lf\n", f_output_grad.abs().sum().item<double>());
//            std::printf("BatchGPUAggregateDst output grad sum: %.4lf\n", f_input_grad.abs().sum().item<double>());
            return f_input_grad;
        }


    };

    class PushDownCPUSrcDstScatterOp : public ntsGraphOp{
    public:
        SampledSubgraph* subgraphs;
        int layer=0;

        PushDownCPUSrcDstScatterOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_)
                : ntsGraphOp(graph_)  {
            subgraphs = subgraphs_;
            layer = layer_;
        }
        NtsVar forward(NtsVar &f_input){
            int feature_size = f_input.size(1);
            assert(subgraphs->sampled_sgs[layer]->e_size > 0);
            NtsVar f_output=graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->e_size,
                                                       2*feature_size},torch::DeviceType::CPU);
            ValueType *f_input_buffer =
                    graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
            ValueType *f_output_buffer =
                    graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
//            print_tensor_size(f_input);
//            std::printf("dst size: %d, src size: %d, edge size: %d\n", subgraphs->sampled_sgs[layer]->v_size,
//                        subgraphs->sampled_sgs[layer]->src_size, subgraphs->sampled_sgs[layer]->e_size);
            assert(layer == 0);
            this->subgraphs->compute_one_layer(
                    [&](VertexId local_dst, std::vector<VertexId>& column_offset,
                        std::vector<VertexId>& row_indices){
                        VertexId src_start=column_offset[local_dst];
                        VertexId src_end=column_offset[local_dst+1];
                        VertexId dst=subgraphs->sampled_sgs[layer]->dst_local_id[local_dst];
                        assert(dst < subgraphs->sampled_sgs[layer]->src_size);

                        for(VertexId src_offset=src_start;
                            src_offset<src_end;src_offset++){
                            VertexId local_src=row_indices[src_offset];
//                            VertexId src=subgraphs->sampled_sgs[layer]->src()[local_src];
                            nts_copy(f_output_buffer, src_offset * 2, f_input_buffer, local_src, feature_size, 1);
                            nts_copy(f_output_buffer, src_offset * 2 + 1, f_input_buffer, dst, feature_size, 1);
                        }
                    },
                    layer
            );
//            graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//                    [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//                        // iterate the incoming edge for vtx
//                        for (long eid = subgraph->column_offset[vtx];
//                             eid < subgraph->column_offset[vtx + 1]; eid++) {
//                            VertexId src = subgraph->row_indices[eid];
//                            nts_copy(f_output_buffer, eid * 2, f_input_buffer, src, feature_size,1);
//                            nts_copy(f_output_buffer, eid * 2 + 1, f_input_buffer, vtx, feature_size,1);
//                        }
//                    },
//                    subgraphs->sampled_sgs[layer], feature_size, active_);
            return f_output;
        }

        NtsVar forward(NtsVar &f_input,std::vector<VertexId> cacheflag){
            NtsVar f_output = graph_->Nts->NewKeyTensor(f_input, torch::DeviceType::CPU);
            return f_output;
        }
        NtsVar backward(NtsVar &f_output_grad){
            int feature_size=f_output_grad.size(1);
            assert(feature_size%2==0);
            // 根据下面的函数应该是累加到src上面
            NtsVar f_input_grad=graph_->Nts->NewLeafTensor({subgraphs->sampled_sgs[layer]->src_size,
                                                            feature_size/2},torch::DeviceType::CPU);

            return f_input_grad;
        }

    };

    class PushDownEdgeSoftMax : public ntsGraphOp{
    public:
        SampledSubgraph* subgraphs;
        int layer=0;

        PushDownEdgeSoftMax(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_)
                : ntsGraphOp(graph_)  {
            subgraphs = subgraphs_;
            layer = layer_;
        }
        NtsVar forward(NtsVar &f_input){
            int feature_size = f_input.size(1);
            assert(subgraphs->sampled_sgs[layer]->e_size > 0);
            NtsVar f_output=graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->e_size,
                                                       feature_size},torch::DeviceType::CPU);
            ValueType *f_input_buffer =
                    graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
            ValueType *f_output_buffer =
                    graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);

            assert(layer == 0);
            this->subgraphs->compute_one_layer(
                    [&](VertexId local_dst, std::vector<VertexId>& column_offset,
                        std::vector<VertexId>& row_indices){
                        VertexId src_start=column_offset[local_dst];
                        VertexId src_end=column_offset[local_dst+1];

                        NtsVar d = f_input.slice(0, src_start, src_end, 1);
                        ValueType *d_buffer = graph_->Nts->getWritableBuffer(d, torch::DeviceType::CPU);
                        nts_copy(f_output_buffer, src_start, d_buffer, 0, feature_size, (src_end - src_start));
                    },
                    layer
            );
//            graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//                    [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//                        // iterate the incoming edge for vtx
//                        for (long eid = subgraph->column_offset[vtx];
//                             eid < subgraph->column_offset[vtx + 1]; eid++) {
//                            VertexId src = subgraph->row_indices[eid];
//                            nts_copy(f_output_buffer, eid * 2, f_input_buffer, src, feature_size,1);
//                            nts_copy(f_output_buffer, eid * 2 + 1, f_input_buffer, vtx, feature_size,1);
//                        }
//                    },
//                    subgraphs->sampled_sgs[layer], feature_size, active_);
            return f_output;
        }

        NtsVar forward(NtsVar &f_input,std::vector<VertexId> cacheflag){
            NtsVar f_output = graph_->Nts->NewKeyTensor(f_input, torch::DeviceType::CPU);
            return f_output;
        }
        NtsVar backward(NtsVar &f_output_grad){
            int feature_size=f_output_grad.size(1);
            assert(feature_size%2==0);
            // 根据下面的函数应该是累加到src上面
            NtsVar f_input_grad=graph_->Nts->NewLeafTensor({subgraphs->sampled_sgs[layer]->src_size,
                                                            feature_size/2},torch::DeviceType::CPU);

            return f_input_grad;
        }

    };


    class PushDownCPUDstAggregateOp : public ntsGraphOp{
    public:
        SampledSubgraph* subgraphs;
        int layer=0;

        PushDownCPUDstAggregateOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_)
                : ntsGraphOp(graph_)  {
            subgraphs = subgraphs_;
            layer = layer_;
        }
        NtsVar forward(NtsVar &f_input){
            int feature_size = f_input.size(1);
            assert(subgraphs->sampled_sgs[layer]->e_size > 0);
            NtsVar f_output=graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->v_size,
                                                       feature_size},torch::DeviceType::CPU);
            ValueType *f_input_buffer =
                    graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
            ValueType *f_output_buffer =
                    graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);

            assert(layer == 0);
            this->subgraphs->compute_one_layer(
                    [&](VertexId vtx, std::vector<VertexId>& column_offset,
                        std::vector<VertexId>& row_indices){

                        for(long eid = column_offset[vtx]; eid < column_offset[vtx + 1]; eid++) {
                            nts_acc(f_input_buffer + (feature_size * eid),
                                    f_output_buffer + vtx * feature_size,
                                    feature_size);
                        }
                    },
                    layer
            );
//            graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//                    [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//                        // iterate the incoming edge for vtx
//                        for (long eid = subgraph->column_offset[vtx];
//                             eid < subgraph->column_offset[vtx + 1]; eid++) {
//                            VertexId src = subgraph->row_indices[eid];
//                            nts_copy(f_output_buffer, eid * 2, f_input_buffer, src, feature_size,1);
//                            nts_copy(f_output_buffer, eid * 2 + 1, f_input_buffer, vtx, feature_size,1);
//                        }
//                    },
//                    subgraphs->sampled_sgs[layer], feature_size, active_);
            return f_output;
        }

        NtsVar forward(NtsVar &f_input,std::vector<VertexId> cacheflag){
            NtsVar f_output = graph_->Nts->NewKeyTensor(f_input, torch::DeviceType::CPU);
            return f_output;
        }
        NtsVar backward(NtsVar &f_output_grad){
            int feature_size=f_output_grad.size(1);
            // 根据下面的函数应该是累加到src上面
            NtsVar f_input_grad=graph_->Nts->NewLeafTensor({subgraphs->sampled_sgs[layer]->src_size,
                                                            feature_size},torch::DeviceType::CPU);

            return f_input_grad;
        }

    };






//    void AVXMul(float* mat1, float* mat2, float* result, int M, int N, int K)
//    {
//
//
//        __m256 vec_multi_res = _mm256_setzero_ps(); //Initialize vector to zero
//        __m256 vec_mat1 = _mm256_setzero_ps(); //Initialize vector to zero
//        __m256 vec_mat2 = _mm256_setzero_ps(); //Initialize vector to zero
//
//        int i, j, k;
//        for (i = 0; i < M; i++)
//        {
//            for (j = 0; j < N; ++j)
//            {
//                //Stores one element in mat1 and use it in all computations needed before proceeding
//                //Stores as vector to increase computations per cycle
//                vec_mat1 = _mm256_set1_epi32(mat1[i*N +j]);
//
//                for (k = 0; k < K; k += 8)
//                {
//                    vec_mat2 = _mm256_loadu_si256((__m256i*)&mat2[j* K + k]); //Stores row of second matrix (eight in each iteration)
//                    vec_multi_res = _mm256_loadu_si256((__m256i*)&result[i* K + k]); //Loads the result matrix row as a vector
//                    vec_multi_res = _mm256_add_epi32(vec_multi_res ,_mm256_mullo_epi32(vec_mat1, vec_mat2));//Multiplies the vectors and adds to th the result vector
//
//                    _mm256_storeu_si256((__m256i*)&result[i*K + k], vec_multi_res); //Stores the result vector into the result array
//                }
//            }
//        }
//    }

//class SingleCPUSrcScatterOp : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  
//  SingleCPUSrcScatterOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input){
//    int feature_size = f_input.size(1);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_e_num, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);            
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//          nts_copy(f_output_buffer, eid, f_input_buffer, src, feature_size,1);
//        }
//      },
//      subgraphs, feature_size, active_);            
//    return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){
//      int feature_size=f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_v_num, 
//                feature_size},torch::DeviceType::CPU);
//              
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    ValueType *f_output_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//            nts_acc(f_output_grad_buffer + (feature_size * eid),
//                    f_input_grad_buffer + src * feature_size,
//                     feature_size);
//        }
//      },
//      subgraphs, feature_size, active_);
//      return f_input_grad;
//  }    
//
//};
//
//class SingleCPUDstAggregateOp : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  
//  SingleCPUDstAggregateOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input){// input edge  output vertex
//    int feature_size = f_input.size(1);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_v_num, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
//    
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//          nts_acc(f_output_buffer + vtx * feature_size, 
//                  f_input_buffer + eid * feature_size, feature_size);
//        }
//      },
//      subgraphs, feature_size, active_);            
//    return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
//      int feature_size=f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_e_num, 
//                feature_size},torch::DeviceType::CPU);
//              
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    ValueType *f_output_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//            nts_acc(f_input_grad_buffer+ (feature_size * eid),
//                    f_output_grad_buffer + vtx * feature_size,
//                     feature_size);
//        }
//      },
//      subgraphs, feature_size, active_);
//      return f_input_grad;
//  }    
//
//};
//
//class SingleCPUDstAggregateOpMin : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  VertexId* record;
//  
//  SingleCPUDstAggregateOpMin(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  ~SingleCPUDstAggregateOpMin(){
//      delete [] record;
//  }
//  NtsVar forward(NtsVar &f_input){// input edge  output vertex
//    int feature_size = f_input.size(1);
//    
//    record=new VertexId(partitioned_graph_->owned_vertices*feature_size);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_v_num, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
//    
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//          nts_min(f_output_buffer+ vtx * feature_size,
//                   f_input_buffer + eid * feature_size, 
//                    record + vtx * feature_size,
//                  feature_size,eid);
//        }
//      },
//      subgraphs, feature_size, active_);            
//    return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
//      int feature_size=f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_e_num, 
//                feature_size},torch::DeviceType::CPU);
//              
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    ValueType *f_output_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        nts_assign(f_input_grad_buffer, f_output_grad_buffer+feature_size*vtx,
//                record+feature_size*vtx, feature_size); 
////        for (long eid = subgraph->column_offset[vtx];
////             eid < subgraph->column_offset[vtx + 1]; eid++) {
////          VertexId src = subgraph->row_indices[eid];
////          
//////            nts_acc(f_input_grad_buffer+ (feature_size * eid),
//////                    f_output_grad_buffer + vtx * feature_size,
//////                     feature_size);
////        }
//      },
//      subgraphs, feature_size, active_);
//      return f_input_grad;
//  }    
//
//};
//
//class SingleCPUDstAggregateOpMax : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  VertexId* record;
//  
//  SingleCPUDstAggregateOpMax(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  ~SingleCPUDstAggregateOpMax(){
//      delete [] record;
//  }
//  NtsVar forward(NtsVar &f_input){// input edge  output vertex
//    int feature_size = f_input.size(1);
//    
//    record=new VertexId(partitioned_graph_->owned_vertices*feature_size);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_v_num, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
//    
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//          nts_max(f_output_buffer+ vtx * feature_size,
//                   f_input_buffer + eid * feature_size, 
//                    record + vtx * feature_size,
//                  feature_size,eid);
//        }
//      },
//      subgraphs, feature_size, active_);            
//    return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
//      int feature_size=f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_e_num, 
//                feature_size},torch::DeviceType::CPU);
//              
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    ValueType *f_output_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        nts_assign(f_input_grad_buffer, f_output_grad_buffer+feature_size*vtx,
//                record+feature_size*vtx, feature_size); 
////        for (long eid = subgraph->column_offset[vtx];
////             eid < subgraph->column_offset[vtx + 1]; eid++) {
////          VertexId src = subgraph->row_indices[eid];
////          
//////            nts_acc(f_input_grad_buffer+ (feature_size * eid),
//////                    f_output_grad_buffer + vtx * feature_size,
//////                     feature_size);
////        }
//      },
//      subgraphs, feature_size, active_);
//      return f_input_grad;
//  }    
//
//};
//
//
//class SingleEdgeSoftMax : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  NtsVar IntermediateResult;
//  
//  SingleEdgeSoftMax(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input_){// input i_msg  output o_msg
//     //NtsVar f_input_=f_input.detach();
//    int feature_size = f_input_.size(1);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_e_num, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input_, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
//    
//        graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//        [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//          long eid_start = subgraph->column_offset[vtx];
//          long eid_end = subgraph->column_offset[vtx + 1];
//          assert(eid_end <= graph_->edges);
//          assert(eid_start >= 0);
//          NtsVar d = f_input_.slice(0, eid_start, eid_end, 1).softmax(0);
//          ValueType *d_buffer =
//          graph_->Nts->getWritableBuffer(d, torch::DeviceType::CPU);      
//          nts_copy(f_output_buffer, eid_start, d_buffer, 
//                  0, feature_size,(eid_end-eid_start));   
//        },
//        subgraphs, f_input_.size(1), this->active_);
//    
//    IntermediateResult=f_output;
//          
//    return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
//    int feature_size=f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_e_num, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//        [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//          long eid_start = subgraph->column_offset[vtx];
//          long eid_end = subgraph->column_offset[vtx + 1];
//          assert(eid_end <= graph_->edges);
//          assert(eid_start >= 0);
//          NtsVar d   = f_output_grad.slice(0, eid_start, eid_end, 1);
//          NtsVar imr =IntermediateResult.slice(0, eid_start, eid_end, 1);
//          //s4=(s2*s1)-(s2)*(s2.t().mm(s1)); 
//          NtsVar d_o =(imr*d)-imr*(d.t().mm(imr)); 
//          ValueType *d_o_buffer =
//          graph_->Nts->getWritableBuffer(d_o, torch::DeviceType::CPU);
//          nts_copy(f_input_grad_buffer, eid_start, d_o_buffer, 
//                  0, feature_size,(eid_end-eid_start));
//        },
//        subgraphs, f_output_grad.size(1), this->active_);
//      return f_input_grad;
//  }    
//
//};



} // namespace graphop
} // namespace nts

#endif
