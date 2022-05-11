#ifndef NTSGRAPHOP_HPP
#define NTSGRAPHOP_HPP
#include <assert.h>
#include <map>
#include <math.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#include "graph.hpp"
#include "ntsBaseOp.hpp"
#include "input.h"

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

class ForwardCPUfuseOp : public ntsGraphOp {
public:
  std::vector<CSC_segment_pinned *> subgraphs;

  ForwardCPUfuseOp(Graph<Empty> *graph, VertexSubset *active,
                   std::vector<CSC_segment_pinned *> &subgraphs_)
      : ntsGraphOp(graph, active) {
    subgraphs = subgraphs_;
  }
  NtsVar forward(NtsVar &f_input) {
    //f_input = input;
    NtsVar f_output = graph_->Nts->NewKeyTensor(f_input, torch::DeviceType::CPU);
    ValueType *f_input_buffer =
        graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
        graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
    memset(f_output_buffer, 0,
           sizeof(ValueType) * f_input.size(0) * f_input.size(1));
    int feature_size = f_input.size(1);
    graph_->process_edges_forward_decoupled_mutisockets<int, ValueType>(
        [&](VertexId src, int current_send_partition) {
          if (graph_->rtminfo->lock_free) {
            VertexId src_trans = src - graph_->gnnctx->p_v_s;
            // check whether this vertex is necessary to send to
            // current_send_partition
            if (subgraphs[current_send_partition]->get_forward_active(
                    src_trans)) {
              // get the index where we shall place the data
              // and invoke emit_buffer_lock_free to send messages
              VertexId write_index =
                  subgraphs[current_send_partition]
                      ->forward_multisocket_message_index[src_trans];
              graph_->NtsComm->emit_buffer_lock_free(
                  src,
                  f_input_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
                  write_index, feature_size);
            }
          } else {
            // send to mirror directly
            graph_->NtsComm->emit_buffer(
                src,
                f_input_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
                feature_size);
          }
        },
        // sparse slot.
        // accumulate incoming feature for dst
        // recv_id represent the partition ID corresponding to current subgraph
        [&](VertexId dst, CSC_segment_pinned *subgraph,
            MessageBuffer **recv_buffer, std::vector<VertexIndex> &src_index,
            VertexId recv_id) {
          VertexId dst_trans =
              dst - graph_->partition_offset[graph_->partition_id];
          // for every vertex, accumulate the incoming feature though iterating
          // column vertices
          for (long idx = subgraph->column_offset[dst_trans];
               idx < subgraph->column_offset[dst_trans + 1]; idx++) {
            VertexId src = subgraph->row_indices[idx];
            VertexId src_trans = src - graph_->partition_offset[recv_id];
            // fetch input from recv buffer
            // bufferIndex indicate which socket we've placed the data
            // positionIndex indicate the index of the buffer of that socket
            ValueType *local_input =
                (ValueType *)(recv_buffer[src_index[src_trans].bufferIndex]
                                  ->data +
                              graph_->sizeofM<ValueType>(feature_size) *
                                  src_index[src_trans].positionIndex +
                              sizeof(VertexId));
            ValueType *local_output =
                f_output_buffer + dst_trans * feature_size;
            nts_comp(local_output, local_input, nts_norm_degree(graph_, src, dst),
                 feature_size);
          }
        },
        subgraphs, feature_size, active_);
    return f_output;
  }
  NtsVar backward(NtsVar &f_output_grad) {
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor(f_output_grad, torch::DeviceType::CPU);
  ValueType *output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
  ValueType *input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
  memset(input_grad_buffer, 0, sizeof(ValueType) * f_output_grad.size(0) * f_output_grad.size(1));
  // int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
  int feature_size = f_output_grad.size(1);
  ValueType *output_buffer = new ValueType[feature_size * graph_->threads];
  graph_->process_edges_backward_decoupled_multisockets<
      int, ValueType>( // For EACH Vertex
                       // Processing
      [&](VertexId src, VertexAdjList<Empty> outgoing_adj, VertexId thread_id,
          VertexId recv_id, VertexId socketId) { // pull
        ValueType *local_output_buffer =
            output_buffer + feature_size * thread_id;
        memset(local_output_buffer, 0, sizeof(ValueType) * feature_size);
        VertexId src_trans = src - graph_->partition_offset[recv_id];

        // iterate the outgoing vertices and combine the gradients
        for (long d_idx = subgraphs[recv_id]->row_offset[src_trans];
             d_idx < subgraphs[recv_id]->row_offset[src_trans + 1]; d_idx++) {
          VertexId dst = subgraphs[recv_id]->column_indices[d_idx];

          // FIXME: will this work?
          if ((dst < graph_->local_partition_offset[socketId]) ||
              (dst >= graph_->local_partition_offset[socketId + 1])) {
            continue;
          }
          VertexId dst_trans = dst - graph_->gnnctx->p_v_s;
          ValueType *local_input_buffer =
             output_grad_buffer + (dst_trans)*feature_size;
          nts_comp(local_output_buffer, local_input_buffer, nts_norm_degree(graph_,src, dst),
               feature_size);
        }
        if (graph_->rtminfo->lock_free) {
          if (subgraphs[recv_id]->source_active->get_bit(src_trans)) {
            VertexId write_index =
                subgraphs[recv_id]
                    ->backward_multisocket_message_index[src_trans]
                    .vertexSocketPosition[socketId];
            graph_->NtsComm->emit_buffer_lock_free(src, local_output_buffer,
                                                   write_index, feature_size);
          }
        } else {
          graph_->NtsComm->emit_buffer(src, local_output_buffer, feature_size);
        }
      },
      [&](VertexId src, ValueType *msg) {
        nts_acc(input_grad_buffer + (src - graph_->gnnctx->p_v_s) * feature_size, msg, feature_size);
        return 1;
      },
      feature_size, active_);
  delete[] output_buffer;
  return f_input_grad;
  
  }
};

class ForwardGPUfuseOp : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;

  ForwardGPUfuseOp(Graph<Empty> *graph, VertexSubset *active,
                   std::vector<CSC_segment_pinned *> &subgraphs_)
      : ntsGraphOp(graph, active) {
    subgraphs = subgraphs_;
  }
  NtsVar forward(NtsVar &f_input){
        int feature_size = f_input.size(1);
  NtsVar f_input_cpu = f_input.cpu();
  NtsVar f_output=graph_->Nts->NewKeyTensor(f_input,torch::DeviceType::CUDA);
  ValueType *f_input_cpu_buffered = f_input_cpu.accessor<ValueType, 2>().data();

  { // original communication
    graph_->sync_compute_decoupled<int, ValueType>(
        f_input, subgraphs,
        [&](VertexId src) {
          graph_->NtsComm->emit_buffer(
              src, f_input_cpu_buffered + (src - graph_->gnnctx->p_v_s) * feature_size,
              feature_size);
        },
        f_output, feature_size);
  }
  return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){
      int feature_size = f_output_grad.size(1);
      NtsVar f_input_grad=graph_->Nts->NewKeyTensor(f_output_grad,torch::DeviceType::CUDA);
  // if (!selective)
  {
    graph_->compute_sync_decoupled<int, ValueType>(
        f_output_grad, subgraphs,
        [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { // pull
          graph_->NtsComm->emit_buffer(
              src, graph_->output_cpu_buffer + (src)*feature_size,
              feature_size);
        },
        f_input_grad, feature_size);
  }
      return f_input_grad;
  }
};

class ForwardSingleGPUfuseOp : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;
  
  ForwardSingleGPUfuseOp(Graph<Empty> *graph, VertexSubset *active,
                   std::vector<CSC_segment_pinned *> &subgraphs_)
      : ntsGraphOp(graph, active) {
    subgraphs = subgraphs_;
  }
  NtsVar forward(NtsVar &f_input){
    int feature_size = f_input.size(1);
    NtsVar f_output=graph_->Nts->NewKeyTensor(f_input,torch::DeviceType::CUDA);
    graph_->forward_single<int, ValueType>(f_input, subgraphs, f_output, feature_size);
    return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){
    int feature_size = f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewKeyTensor(f_output_grad,torch::DeviceType::CUDA);
    graph_->backward_single<int, ValueType>(f_output_grad, subgraphs, 
            f_input_grad, feature_size);
      return f_input_grad;
  }    

};

class SingleCPUSrcDstScatterOp : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;
  
  SingleCPUSrcDstScatterOp(Graph<Empty> *graph, VertexSubset *active,
                   std::vector<CSC_segment_pinned *> &subgraphs_)
      : ntsGraphOp(graph, active) {
    subgraphs = subgraphs_;
  }
  NtsVar forward(NtsVar &f_input){
    int feature_size = f_input.size(1);
    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_e_num, 
                2*feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);            
  graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
          nts_copy(f_output_buffer, eid * 2, f_input_buffer, src, feature_size);
          nts_copy(f_output_buffer, eid * 2 + 1, f_input_buffer, vtx, feature_size);
        }
      },
      subgraphs, feature_size, active_);            
    return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){
      int feature_size=f_output_grad.size(1);
                     assert(feature_size%2==0); 
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_v_num, 
                feature_size/2},torch::DeviceType::CPU);
              
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
    feature_size/=2;
      graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
            nts_acc(f_input_grad_buffer + src * feature_size,
                    f_output_grad_buffer + (feature_size * eid * 2), feature_size);
            nts_acc(f_input_grad_buffer + vtx * feature_size,
                    f_output_grad_buffer + (feature_size * (eid * 2 + 1)),
                     feature_size);
        }
      },
      subgraphs, feature_size, active_);
      return f_input_grad;
  }    

};

class SingleCPUSrcScatterOp : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;
  
  SingleCPUSrcScatterOp(Graph<Empty> *graph, VertexSubset *active,
                   std::vector<CSC_segment_pinned *> &subgraphs_)
      : ntsGraphOp(graph, active) {
    subgraphs = subgraphs_;
  }
  NtsVar forward(NtsVar &f_input){
    int feature_size = f_input.size(1);
    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_e_num, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);            
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
          nts_copy(f_output_buffer, eid, f_input_buffer, src, feature_size);
//          nts_copy(f_output_buffer, eid * 2, f_input_buffer, src, feature_size);
//          nts_copy(f_output_buffer, eid * 2 + 1, f_input_buffer, vtx, feature_size);
        }
      },
      subgraphs, feature_size, active_);            
    return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){
      int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_v_num, 
                feature_size},torch::DeviceType::CPU);
              
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
            nts_acc(f_output_grad_buffer + (feature_size * eid),
                    f_input_grad_buffer + src * feature_size,
                     feature_size);
        }
      },
      subgraphs, feature_size, active_);
      return f_input_grad;
  }    

};

class SingleCPUDstAggregateOp : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;
  
  SingleCPUDstAggregateOp(Graph<Empty> *graph, VertexSubset *active,
                   std::vector<CSC_segment_pinned *> &subgraphs_)
      : ntsGraphOp(graph, active) {
    subgraphs = subgraphs_;
  }
  NtsVar forward(NtsVar &f_input){// input edge  output vertex
    int feature_size = f_input.size(1);
    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_v_num, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
    
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
          nts_acc(f_output_buffer + vtx * feature_size, 
                  f_input_buffer + eid * feature_size, feature_size);
        }
      },
      subgraphs, feature_size, active_);            
    return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
      int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_e_num, 
                feature_size},torch::DeviceType::CPU);
              
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
            nts_acc(f_input_grad_buffer+ (feature_size * eid),
                    f_output_grad_buffer + src * feature_size,
                     feature_size);
        }
      },
      subgraphs, feature_size, active_);
      return f_input_grad;
  }    

};


} // namespace graphop
} // namespace nts

#endif