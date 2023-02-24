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
#ifndef NTSSINGLEGPUSAMPLEGRAPHOP_HPP
#define NTSSINGLEGPUSAMPLEGRAPHOP_HPP
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
    
//#if CUDA_ENABLE    
class SingleGPUSampleGraphOp : public ntsGraphOp{
public:
  SampledSubgraph *subgraphs;
  int layer;
  Cuda_Stream* cuda_stream;
  SingleGPUSampleGraphOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_,Cuda_Stream* cuda_stream_)
      : ntsGraphOp(graph_) {
    subgraphs = subgraphs_;
    layer = layer_;
    cuda_stream = cuda_stream_;
  }

  NtsVar forward(NtsVar &f_input){
    int feature_size = f_input.size(1);
    // printf("forward feature:%d layer:%d\n",feature_size,layer);
    // NtsVar f_output = subgraphs->forward_embedding[layer];
    // graph_->Nts->ZeroVarMem(f_output);
      // std::printf("forward subgraphs->sampled_sgs[layer]->src_size: %d\n", subgraphs->sampled_sgs[layer]->src_size);
    NtsVar f_output = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->v_size,feature_size},torch::DeviceType::CUDA);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CUDA);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CUDA);
    float* weight_forward=subgraphs->sampled_sgs[layer]->dev_e_w_f();
    VertexId* row_indices=subgraphs->sampled_sgs[layer]->dev_r_i();
    VertexId* column_offset=subgraphs->sampled_sgs[layer]->dev_c_o();
    VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
    VertexId batch_size=subgraphs->sampled_sgs[layer]->v_size;
    
    // std::printf("f_input size(%d, %d), batch size: %d, feature size: %d\n", f_input.size(0), 
    //             f_input.size(1), batch_size, feature_size);
    // std::printf("f_output size: (%d, %d)\n", f_output.size(0), f_output.size(1));
    // std::printf("subgraphs->sampled_sgs[layer]: %d\n", subgraphs->sampled_sgs[layer]->size_dev_dst);
    // // std::printf("test: %d\n", column_offset[batch_size]);
    // std::printf("edge: %d, dev edge: %d\n", edge_size, subgraphs->sampled_sgs[layer]->size_dev_edge);
    // std::printf("layer: %d, src size: %d\n", layer, subgraphs->sampled_sgs[layer]->src_size);
    // NtsVar f_output_tmp = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->v_size,feature_size},torch::DeviceType::CUDA);
    // ValueType *f_output_buffer_tmp =
    //   graph_->Nts->getWritableBuffer(f_output_tmp, torch::DeviceType::CUDA);


    int column_num = subgraphs->sampled_sgs[layer]->src_size;
    if(feature_size<=512 && false){
      cuda_stream->Gather_By_Dst_From_Src_Optim(
        f_input_buffer, f_output_buffer, weight_forward, // data
          row_indices, column_offset, 0, 0, 0, 0,
            edge_size, batch_size,feature_size, true, false);
    }else{
      // cuda_stream->Gather_By_Dst_From_Src(
      //   f_input_buffer, f_output_buffer, weight_forward, // data
      //     row_indices, column_offset, 0, 0, 0, 0,
      //       edge_size, batch_size,feature_size, true, false);
      cuda_stream->Gather_By_Dst_From_Src_Spmm(
        f_input_buffer, f_output_buffer, weight_forward, // data
          row_indices, column_offset, column_num, 0, 0, 0, 0,
            edge_size, batch_size,feature_size, true, false);
    }
     // TODO: toao注释掉
    // cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    // // std::printf("\noutput size: (%d, %d)\n", f_output.size(0), f_output.size(1));
    // // std::cout << "output sum: " << f_output.sum().item<double>() << std::endl;
    // std::printf("real: %lf, compute: %lf\n", f_output_tmp.abs().sum().item<double>(), f_output.abs().sum().item<double>());
    // f_output.is_contiguous();
    
    // f_output.requires_grad_(false);
    // f_output.resize_({feature_size, batch_size});
    // f_output.t_();
    // f_output.requires_grad_(true);
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
    // std::printf("backward subgraphs->sampled_sgs[layer]->src_size: %d\n", subgraphs->sampled_sgs[layer]->src_size);
    NtsVar f_input_grad = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->src_size,feature_size},torch::DeviceType::CUDA);
    ValueType *f_input_grad_buffer =
          graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CUDA);
    ValueType *f_output_grad_buffer =
          graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CUDA);
    VertexId *row_offset = subgraphs->sampled_sgs[layer]->dev_r_o();
    VertexId *column_indices = subgraphs->sampled_sgs[layer]->dev_c_i();
    ValueType *weight_backward = subgraphs->sampled_sgs[layer]->dev_e_w_b();
    VertexId edge_size=subgraphs->sampled_sgs[layer]->e_size;
    VertexId batch_size=subgraphs->sampled_sgs[layer]->src_size;
    // assert(f_output_grad.size(0) == batch_size);
    // std::printf("output_grad size(%d, %d), batch size: %d, feature size: %d\n", f_output_grad.size(0), 
    //             f_output_grad.size(1), batch_size, feature_size);
//    std::printf("param grad size: (%d, %d)\n", f_output_grad.size(0), f_output_grad.size(1));
//     std::printf("return grad size: (%d, %d)\n", f_input_grad.size(0), f_input_grad.size(1));
    int column_num = f_output_grad.size(0);
    if (feature_size <= 512 && false) {
      cuda_stream->Gather_By_Src_From_Dst_Optim(
          f_output_grad_buffer, f_input_grad_buffer,
          weight_backward,
          row_offset, // graph
          column_indices, 0,0,0,0,edge_size,
          batch_size, feature_size,
          true,false);
    } else {
      // cuda_stream->Gather_By_Src_From_Dst(
      //     f_output_grad_buffer, f_input_grad_buffer,
      //     weight_backward,
      //     row_offset, // graph
      //     column_indices, 0,0,0,0,edge_size,
      //     batch_size, feature_size,
      //     true,false);
      cuda_stream->Gather_By_Src_From_Dst_Spmm(
          f_output_grad_buffer, f_input_grad_buffer,
          weight_backward,
          row_offset, // graph
          column_indices, column_num, 0,0,0,0,edge_size,
          batch_size, feature_size,
          true,false);
    }
     // TODO: toao注释掉
    // cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    return f_input_grad;
  }    

};

class SingleGPUAllSampleGraphOp : public ntsGraphOp{
public:
  SampledSubgraph *subgraphs;
  int layer;
  Cuda_Stream* cuda_stream;
  SingleGPUAllSampleGraphOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_,Cuda_Stream* cuda_stream_)
      : ntsGraphOp(graph_) {
    subgraphs = subgraphs_;
    layer = layer_;
    cuda_stream = cuda_stream_;
  }

  NtsVar forward(NtsVar &f_input){
    int feature_size = f_input.size(1);
    // printf("forward feature:%d layer:%d\n",feature_size,layer);
    // NtsVar f_output = subgraphs->forward_embedding[layer];
    // graph_->Nts->ZeroVarMem(f_output);
    NtsVar f_output = graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->v_size,feature_size},torch::DeviceType::CUDA);
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
    if(feature_size<=512 && false){
      cuda_stream->Gather_By_Dst_From_Src_Optim(
        f_input_buffer, f_output_buffer, weight_forward, // data
          row_indices, column_offset, 0, 0, 0, 0,
            edge_size, batch_size,feature_size, true, false);
    }else{
//        cuda_stream->Gather_By_Dst_From_Src(
//                f_input_buffer, f_output_buffer, weight_forward, // data
//                row_indices, column_offset,0, 0, 0, 0,
//                edge_size, batch_size,feature_size, true, false);
      cuda_stream->Gather_By_Dst_From_Src_Spmm(
        f_input_buffer, f_output_buffer, weight_forward, // data
          row_indices, column_offset, column_num,0, 0, 0, 0,
            edge_size, batch_size,feature_size, true, false);
    }

     
//    cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
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


//    std::printf("param grad size: (%d, %d)\n", f_output_grad.size(0), f_output_grad.size(1));
//     std::printf("return grad size: (%d, %d)\n", f_input_grad.size(0), f_input_grad.size(1));
//     std::printf("batch size: %d, column num: %d\n", batch_size, f_input_grad.size(0));

      assert(f_output_grad.size(0) == batch_size);
      int column_num = f_input_grad.size(0);
    if (feature_size <= 512 && false) {
       cuda_stream->Gather_By_Dst_From_Src_Optim(f_output_grad_buffer, f_input_grad_buffer, weight_backward,
                                                  row_indices, column_offset, 0, 0, 0, 0, edge_size, batch_size,
                                                  feature_size, true, false);
    } else {
//        cuda_stream->Push_From_Dst_To_Src(f_output_grad_buffer, f_input_grad_buffer, weight_backward, row_indices,
//                                                column_offset, // graph
//                                                0, 0, 0, 0, edge_size, batch_size, feature_size, true, false);
      cuda_stream->Push_From_Dst_To_Src_Spmm(f_output_grad_buffer, f_input_grad_buffer, weight_backward, row_indices,
                                            column_offset, column_num, // graph
                                            0, 0, 0, 0, edge_size, batch_size, feature_size, true, false);
    }
    return f_input_grad;
  }    

};
//#endif



} // namespace graphop
} // namespace nts

#endif
