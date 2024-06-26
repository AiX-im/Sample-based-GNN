/*
Copyright (c) 2022-2023 xin ai, Northeastern University

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
#ifndef NTSSINGLEGPUSHAREGRAPHOP_HPP
#define NTSSINGLEGPUSHAREGRAPHOP_HPP
#include <assert.h>
#include <map>
#include <math.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#include "core/graph.hpp"
#include "core/ntsBaseOp.hpp"
#include "core/PartitionedGraph.hpp"

namespace nts {
namespace op {

//class ntsGraphOp {
//public:
//  Graph<Empty> *graph_;
//  VertexSubset *active_;
//  ntsGraphOp() { ; }
//  ntsGraphOp(Graph<Empty> *graph, VertexSubset *active) {
//    graph_ = graph;
//    active_ = active;
//  }
//  virtual NtsVar &forward(NtsVar &input) = 0;
//  virtual NtsVar backward(NtsVar &output_grad) = 0;
//};
    
#if CUDA_ENABLE    
class SingleGPUShareGraphOp : public ntsGraphOp{
public:
  Cuda_Stream* cuda_stream;
  SampledSubgraph *subgraphs;
  FastSampler *sampler;
  int layer;
  SingleGPUShareGraphOp(FastSampler *sampler_, SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_)
      : ntsGraphOp(graph_) {
    subgraphs = subgraphs_;
    layer = layer_;
    sampler = sampler_;
    cuda_stream=new Cuda_Stream();
  }
  NtsVar forward(NtsVar &f_input){
    int feature_size = f_input.size(1);
    // printf("forward feature:%d layer:%d\n",feature_size,layer);
    NtsVar f_output=graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->v_size,
                feature_size}, torch::DeviceType::CUDA);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CUDA);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CUDA);
    float* weight_forward=subgraphs->sampled_sgs[layer]->dev_e_w_f();
    VertexId* row_indices=subgraphs->sampled_sgs[layer]->dev_r_i();
    VertexId* column_offset=subgraphs->sampled_sgs[layer]->dev_c_o();
    VertexId* destination = subgraphs->sampled_sgs[layer]->dev_destination;
    VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
    VertexId batch_size=subgraphs->sampled_sgs[layer]->v_size;
    cuda_stream->Gather_By_Dst_From_Src(
        f_input_buffer, f_output_buffer, weight_forward, // data
          row_indices, column_offset, 0, 0, 0, 0,
            edge_size, batch_size,feature_size, true, false);
    cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
      return f_output;
  }
  NtsVar forward(NtsVar &f_input,std::vector<VertexId> cacheflag){        
      NtsVar f_output = graph_->Nts->NewKeyTensor(f_input, torch::DeviceType::CPU); 
    return f_output;
  }
  NtsVar backward(NtsVar &f_output_grad){
    int feature_size = f_output_grad.size(1);
    // printf("backward feature:%d layer:%d\n",feature_size,layer);
    NtsVar f_input_grad=graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->src().size(),feature_size},torch::DeviceType::CUDA);
    ValueType *f_input_grad_buffer =
          graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CUDA);
    ValueType *f_output_grad_buffer =
          graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CUDA);
    VertexId *row_offset = subgraphs->sampled_sgs[layer]->dev_r_o();
    VertexId *column_indices = subgraphs->sampled_sgs[layer]->dev_c_i();
    ValueType *weight_backward = subgraphs->sampled_sgs[layer]->dev_e_w_b();
    VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
    VertexId batch_size=subgraphs->sampled_sgs[layer]->src_size;
    if (feature_size <= 512 && false) {
      cuda_stream->Gather_By_Src_From_Dst_Optim(
          f_output_grad_buffer, f_input_grad_buffer,
          weight_backward,
          row_offset, // graph
          column_indices, 0,0,0,0,edge_size,
          batch_size, feature_size,
          true,false);
    } else {
      cuda_stream->Gather_By_Src_From_Dst(
          f_output_grad_buffer, f_input_grad_buffer,
          weight_backward,
          row_offset, // graph
          column_indices, 0,0,0,0,edge_size,
          batch_size, feature_size,
          true,false);
    }
    cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    return f_input_grad;
  }    

};
#endif



} // namespace graphop
} // namespace nts

#endif
