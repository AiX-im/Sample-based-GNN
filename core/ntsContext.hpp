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
#ifndef NTSOPS_HPP
#define NTSOPS_HPP
#include <stack>
#include "ntsGraphOp.hpp"
#include<type_traits>
namespace nts {

namespace ctx {

typedef uint32_t OpType;

const OpType NNOP = 0;
const OpType GRAPHOP = 1;
const OpType SELFNNOP = 2;

class IOTensorId{
public:
    IOTensorId(long o_id_,long i_id1_,long i_id2_){
        o_id=o_id_;
        i_id1=i_id1_;
        i_id2=i_id2_;
    }
    IOTensorId(long o_id_,long i_id1_){
        o_id=o_id_;
        i_id1=i_id1_;
    }
    void updateOutput(long o_id_){
       o_id=o_id_; 
    }
    long o_id;
    long i_id1;
    long i_id2;
};
class ntsOperator{
public:
    ntsOperator(){
        
    }
    ntsOperator(nts::op::ntsGraphOp* op_,OpType op_t_){
        assert(GRAPHOP==op_t_);
        op=op_;
        op_t=op_t_;
    }
    ntsOperator(nts::op::ntsNNBaseOp* op_,OpType op_t_){
        assert(SELFNNOP==op_t_);
        opn=op_;
        op_t=op_t_;
    }
    ntsOperator(OpType op_t_){
        assert(NNOP==op_t_);
        op_t=op_t_;
    }
//    ntsOperator(OpType op_t_){
//        assert(CATOP==op_t_);
//        op_t=op_t_;
//    }
    nts::op::ntsGraphOp* get_graphop(){
        return op;
    }
    nts::op::ntsNNBaseOp* get_nnop(){
        return opn;
    }
    OpType get_op_T(){
        return op_t;
    }
    nts::op::ntsGraphOp* op;
    nts::op::ntsNNBaseOp* opn;
    OpType op_t;
};
/**
 * @brief
 * since GNN operation is just iteration of graph operation and NN operation.
 * so we can simply use a chain to represent GNN operation, which can reduce
 * system complexity greatly.
 * you can also regard it as the NN operation splited by graph operation.
 * And as the extention of auto diff library, we will provide backward
 * computation for graph operation. And thus, the computation path of GNN is
 * constructed.
 */
class NtsContext {
public:
  NtsContext(){
  std::stack<OpType>().swap(op);
  std::stack<NtsVar>().swap(output);
  std::stack<NtsVar>().swap(input);
  std::stack<ntsOperator>().swap(ntsOp);
  output_grad.clear();
  iot_id.clear();
  count = 0;
  training = true;
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

  template <typename GOPT>
  NtsVar runGraphOp(PartitionedGraph* partitioned_graph, VertexSubset *active,
        NtsVar &f_input){//graph op
    static_assert(std::is_base_of<nts::op::ntsGraphOp,GOPT>::value,
                "template must be a type of graph op!");

    nts::op::ntsGraphOp * curr=new GOPT(partitioned_graph,active);
    NtsVar f_output=curr->forward(f_input);
    if (this->training == true) {
      NtsVar ig;
      op.push(GRAPHOP);
      output.push(f_output);
      input.push(f_input);
      ntsOp.push(ntsOperator(curr,GRAPHOP));
      iot_id.push_back(IOTensorId((long)(f_output.data_ptr()),(long)(f_input.data_ptr())));
      // pre-alloc space to save graident
      output_grad.push_back(ig);
      count++;
    }
    return f_output;
}

  template <typename GOPT>
  NtsVar runGraphOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_,
        NtsVar &f_input){//graph op
      
    static_assert(std::is_base_of<nts::op::ntsGraphOp,GOPT>::value,
                "template must be a type of graph op!");
    nts::op::ntsGraphOp * curr=new GOPT(subgraphs_,graph_, layer_);
    NtsVar f_output=curr->forward(f_input);
    if (this->training == true) {
      NtsVar ig;
      op.push(GRAPHOP);
      output.push(f_output);
      input.push(f_input);
      ntsOp.push(ntsOperator(curr,GRAPHOP));
      iot_id.push_back(IOTensorId((long)(f_output.data_ptr()),(long)(f_input.data_ptr())));
      // pre-alloc space to save graident
      output_grad.push_back(ig);
      count++;
    }
    return f_output;
}

    template <typename GOPT>
    NtsVar runGraphOpNoBackward(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_,
                      NtsVar &f_input, int batch_start, int batch_end){//graph op

        static_assert(std::is_base_of<nts::op::ntsGraphOp,GOPT>::value,
                      "template must be a type of graph op!");
        nts::op::ntsGraphOp * curr=new GOPT(subgraphs_,graph_, layer_, batch_start, batch_end);
        NtsVar f_output=curr->forward(f_input);
        return f_output;
    }

    template <typename GOPT>
  NtsVar runGraphOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_,
        NtsVar &f_input,Cuda_Stream* cuda_stream){//graph op
      
    static_assert(std::is_base_of<nts::op::ntsGraphOp,GOPT>::value,
                "template must be a type of graph op!");
    nts::op::ntsGraphOp * curr=new GOPT(subgraphs_,graph_, layer_,cuda_stream);
    NtsVar f_output=curr->forward(f_input);
    if (this->training == true) {
      NtsVar ig;
      op.push(GRAPHOP);
      output.push(f_output);
      input.push(f_input);
      ntsOp.push(ntsOperator(curr,GRAPHOP));
      iot_id.push_back(IOTensorId((long)(f_output.data_ptr()),(long)(f_input.data_ptr())));
      // pre-alloc space to save graident
      output_grad.push_back(ig);
      count++;
    }
    return f_output;
}  

  template <typename GOPT>
  NtsVar runGraphOp(FastSampler *sampler,SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_,
        NtsVar &f_input){//graph op
      
    static_assert(std::is_base_of<nts::op::ntsGraphOp,GOPT>::value,
                "template must be a type of graph op!");
    nts::op::ntsGraphOp * curr=new GOPT(sampler,subgraphs_,graph_, layer_);
    NtsVar f_output=curr->forward(f_input);
    if (this->training == true) {
      NtsVar ig;
      op.push(GRAPHOP);
      output.push(f_output);
      input.push(f_input);
      ntsOp.push(ntsOperator(curr,GRAPHOP));
      iot_id.push_back(IOTensorId((long)(f_output.data_ptr()),(long)(f_input.data_ptr())));
      // pre-alloc space to save graident
      output_grad.push_back(ig);
      count++;
    }
    return f_output;
} 

  template <typename GOPT>
  NtsVar runGraphOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,ValueType *f_embedding, int layer_,
        NtsVar &f_input){//graph op
      
    static_assert(std::is_base_of<nts::op::ntsGraphOp,GOPT>::value,
                "template must be a type of graph op!");
    nts::op::ntsGraphOp * curr=new GOPT(subgraphs_,graph_,f_embedding,layer_);
    NtsVar f_output=curr->forward(f_input);
    if (this->training == true) {
      NtsVar ig;
      op.push(GRAPHOP);
      output.push(f_output);
      input.push(f_input);
      ntsOp.push(ntsOperator(curr,GRAPHOP));
      iot_id.push_back(IOTensorId((long)(f_output.data_ptr()),(long)(f_input.data_ptr())));
      // pre-alloc space to save graident
      output_grad.push_back(ig);
      count++;
    }
    return f_output;
}  

  template <typename GOPT>
  void PushdownGraphOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_,
        NtsVar &f_output, NtsVar &f_input){//graph op
      
    static_assert(std::is_base_of<nts::op::ntsGraphOp,GOPT>::value,
                "template must be a type of graph op!");
    nts::op::ntsGraphOp * curr=new GOPT(subgraphs_,graph_, layer_);
    // NtsVar f_output_t = curr->forward(f_input);
    if (this->training == true) {
      NtsVar ig;
      op.push(GRAPHOP);
      output.push(f_output);
      input.push(f_input);
      ntsOp.push(ntsOperator(curr,GRAPHOP));
      iot_id.push_back(IOTensorId((long)(f_output.data_ptr()),(long)(f_input.data_ptr())));
      // pre-alloc space to save graident
      output_grad.push_back(ig);
      count++;
    }
}  

template <typename NOPT>
  NtsVar runSelfNNOp(std::function<NtsVar(NtsVar &, int)> vertexforward,NtsVar& f_input,int layer_){//graph op
      
    static_assert(std::is_base_of<nts::op::ntsNNBaseOp,NOPT>::value,
                "template must be a type of graph op!");
    
    nts::op::ntsNNBaseOp * curr=new NOPT([&](NtsVar v_tensor,int layer_){
        return vertexforward(v_tensor,layer_);
    },layer_);
    NtsVar f_output=curr->forward(f_input);
    if (this->training == true) {
      NtsVar ig;
      op.push(SELFNNOP);
      output.push(f_output);
      input.push(f_input);
      ntsOp.push(ntsOperator(curr,SELFNNOP));
      iot_id.push_back(IOTensorId((long)(f_output.data_ptr()),(long)(f_input.data_ptr())));
      // pre-alloc space to save graident
      output_grad.push_back(ig);
      count++;
    }
    return f_output;
}  
  
  template <typename GOPT>
  NtsVar runGraphOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_,
        NtsVar &f_input,std::vector<VertexId> cacheflag){//graph op
      
    static_assert(std::is_base_of<nts::op::ntsGraphOp,GOPT>::value,
                "template must be a type of graph op!");
    
    nts::op::ntsGraphOp * curr=new GOPT(subgraphs_,graph_,layer_);
    NtsVar f_output=curr->forward(f_input,cacheflag);
    if (this->training == true) {
      NtsVar ig;
      op.push(GRAPHOP);
      output.push(f_output);
      input.push(f_input);
      ntsOp.push(ntsOperator(curr,GRAPHOP));
      iot_id.push_back(IOTensorId((long)(f_output.data_ptr()),(long)(f_input.data_ptr())));
      // pre-alloc space to save graident
      output_grad.push_back(ig);
      count++;
      if(layer_ == 1)
        cachecount = count + 1;
    }
    return f_output;
}  

 NtsVar runVertexForward(std::function<NtsVar(NtsVar &, NtsVar &)> vertexforward,
            NtsVar &nbr_input,NtsVar &vtx_input){//NNOP
//     LOG_INFO("call run vertex forward");
    NtsVar f_output=vertexforward(nbr_input,vtx_input);
    if (this->training == true) { 
      appendNNOp(nbr_input, f_output);
    }
//    printf("tese %ld\n",(long)(&f_output));
    return f_output;
}
 NtsVar runVertexForward(std::function<NtsVar(NtsVar &)> vertexforward,
            NtsVar &nbr_input){//NNOP
//     LOG_INFO("call run vertex forward");
    NtsVar f_output=vertexforward(nbr_input);
    if (this->training == true) { 
      appendNNOp(nbr_input, f_output);
    }
    return f_output;
}
 
 NtsVar runEdgeForward(std::function<NtsVar(NtsVar &)> edgeforward,
            NtsVar &edge_input){//NNOP
//     LOG_INFO("call run vertex forward");
    NtsVar f_output=edgeforward(edge_input);
    if (this->training == true) { 
      appendNNOp(edge_input, f_output);
    }
    return f_output;
} 
  
  void appendNNOp(NtsVar &input_t, NtsVar &output_t){
    assert(this->training);
    NtsVar ig;

    // we will chain the NNOP together, because torch lib will handle the backward propagation
    // when there is no graph operation
    if (count > 0 && op.top() == NNOP) {
        output.pop();
        output.push(output_t);
        iot_id[iot_id.size()-1].updateOutput((long)(output_t.data_ptr()));
    } else {
        op.push(NNOP);
        output.push(output_t);
        input.push(input_t);
        ntsOp.push(ntsOperator(NNOP));
        iot_id.push_back(IOTensorId((long)(output_t.data_ptr()),(long)(input_t.data_ptr())));
        // pre-alloc space to save graident
        output_grad.push_back(ig);
        count++;
    }
  }
 
  void reset(){
    assert(count<=1);
    if(count==1&&ntsOp.top().op_t==GRAPHOP){
        delete ntsOp.top().op;
    }
    count = 0;
    std::stack<OpType>().swap(op);
    std::stack<NtsVar>().swap(output);
    std::stack<NtsVar>().swap(input);
    std::stack<ntsOperator>().swap(ntsOp);
    output_grad.clear();
    iot_id.clear();
}
  void pop_one_op(){
    if(ntsOp.top().op_t==GRAPHOP){
        delete ntsOp.top().op;
    }
    op.pop();
    output.pop();
    input.pop();
    ntsOp.pop();
    count--;
  }
  void self_backward(bool retain_graph = true){
    assert(this->training);
    output.top().backward(torch::ones_like(output.top()), retain_graph);
    output_grad[top_idx()-1]= input.top().grad();// grad of loss
    pop_one_op();
    //LOG_INFO("FINISH LOSS");
      while (count > 1 || (count == 1 && NNOP == op.top())) {
    // NNOP means we are using torch lib to do the forward computation
    // thus we can use auto diff framework in libtorch
         
    if (GRAPHOP == op.top()) {
      double graph_time = 0;
      graph_time -= get_time();
      int preop_id=top_idx(); 
      if(output_grad[top_idx()].dim()<2){
          output_grad[top_idx()]=output.top().grad();
      }//determine o grad
      for(preop_id=top_idx();preop_id>=0;preop_id--){
          if(iot_id[preop_id].o_id==iot_id[top_idx()].i_id1)
              break;
      }// where to write i grad
      //LOG_INFO("input id %ld %d %d",preop_id,top_idx(),output_grad[top_idx()].dim());
      output_grad[preop_id]=ntsOp.top().get_graphop()->backward(output_grad[top_idx()]);

      // printf("output_grad size:%d\n",output_grad[preop_id].size(0));
//stable      
//      output_grad[top_idx()-1]=ntsOp.top().op->backward(output_grad[top_idx()]);
      pop_one_op();
      graph_time += get_time();
      //printf("#graph_time=%lf(s)\n", graph_time * 10);
    }else if (SELFNNOP == op.top()) { // test
      int preop_id=top_idx(); 
      if(output_grad[top_idx()].dim()<2){
          output_grad[top_idx()]=output.top().grad();
      }//determine o grad
      for(preop_id=top_idx();preop_id>=0;preop_id--){
          if(iot_id[preop_id].o_id==iot_id[top_idx()].i_id1)
              break;
      }// where to write i grad
      output_grad[preop_id]=ntsOp.top().get_nnop()->backward(output_grad[top_idx()]);
//stable      
//      output_grad[top_idx()-1]=ntsOp.top().op->backward(output_grad[top_idx()]);
      pop_one_op();
    }  else if (NNOP == op.top()) {// directly use pytorch
      if(output_grad[top_idx()].dim()<2){
        //printf("1\n");
          output_grad[top_idx()]=output.top().grad();
      }//determine o grad
      if(output_grad[top_idx()].dim()>1){
        //printf("2\n");
        // std::printf("top_idx(): %d\n",top_idx());
        // std::printf("output_grad[top_idx()].size(1):%d, output.top().size(1):%d\n", output_grad[top_idx()].size(1), output.top().size(1));
        assert(output_grad[top_idx()].size(1)==output.top().size(1));
        //  printf("output_grad[top_idx()].size(0):%d output.top().size(0):%d\n",output_grad[top_idx()].size(0),output.top().size(0));
        assert(output_grad[top_idx()].size(0)==output.top().size(0)); 
        output.top().backward(output_grad[top_idx()], retain_graph);
      }

      pop_one_op();
    //LOG_INFO("FINISH NN OP");
    } else {
      LOG_INFO("NOT SUPPORT OP");
      assert(true);
    }
  }
    reset();  
  }

    void Grad_accumulate(NtsVar &Embedding_Grad,
                         ValueType *dev_share_grad,
                         VertexId* cacheflag,
                         VertexId* cachemap,
                         VertexId_CUDA *destination_vertex,
                         VertexId_CUDA vertex_size,
                         Graph<Empty> *graph_,
                         Cuda_Stream *cudaStream){
          ValueType * dev_grad_buffer = graph_->Nts->getWritableBuffer(Embedding_Grad, torch::DeviceType::CUDA);
         cudaStream->dev_Grad_accumulate(dev_grad_buffer,
                                dev_share_grad,
                                cacheflag,
                                cachemap,
                                Embedding_Grad.size(1),
                                destination_vertex,
                                vertex_size);
        }

    void self_backward_cache(bool retain_graph, 
                            VertexId* cacheflag,
                            VertexId* cachemap,
                            VertexId_CUDA *destination_vertex,
                            VertexId_CUDA vertex_size,
                            Graph<Empty> *graph_,
                            Cuda_Stream * cudaStream,
                            ValueType *dev_share_grad){
      assert(this->training);
    output.top().backward(torch::ones_like(output.top()), retain_graph);
    output_grad[top_idx()-1]= input.top().grad();// grad of loss
    pop_one_op();
    //LOG_INFO("FINISH LOSS");
      while (count > 1 || (count == 1 && NNOP == op.top())) {
    // NNOP means we are using torch lib to do the forward computation
    // thus we can use auto diff framework in libtorch
         
    if (GRAPHOP == op.top()) {
      int preop_id=top_idx(); 
      if(output_grad[top_idx()].dim()<2){
          output_grad[top_idx()]=output.top().grad();
      }//determine o grad
      for(preop_id=top_idx();preop_id>=0;preop_id--){
          if(iot_id[preop_id].o_id==iot_id[top_idx()].i_id1)
              break;
      }// where to write i grad
      output_grad[preop_id]=ntsOp.top().get_graphop()->backward(output_grad[top_idx()]);
      //LOG_INFO("input id %ld %d %d",preop_id,top_idx(),output_grad[preop_id].dim());
//stable      
//      output_grad[top_idx()-1]=ntsOp.top().op->backward(output_grad[top_idx()]);
      pop_one_op();
    }else if (SELFNNOP == op.top()) { // test
      int preop_id=top_idx(); 
      if(output_grad[top_idx()].dim()<2){
          output_grad[top_idx()]=output.top().grad();
      }//determine o grad
      for(preop_id=top_idx();preop_id>=0;preop_id--){
          if(iot_id[preop_id].o_id==iot_id[top_idx()].i_id1)
              break;
      }// where to write i grad
      output_grad[preop_id]=ntsOp.top().get_nnop()->backward(output_grad[top_idx()]);
//stable      
//      output_grad[top_idx()-1]=ntsOp.top().op->backward(output_grad[top_idx()]);
      pop_one_op();
    }  else if (NNOP == op.top()) {// directly use pytorch

      if(output_grad[top_idx()].dim()<2){
          output_grad[top_idx()]=output.top().grad();
      }//determine o grad
      assert(output_grad[top_idx()].size(1)==output.top().size(1));
      assert(output_grad[top_idx()].size(0)==output.top().size(0)); 
      
      //accumulative grad back to CPU
      if(output_grad[top_idx()].size(1) == graph_->gnnctx->layer_size[1]) { //判断是否为bottom layer nn
          Grad_accumulate(output_grad[top_idx()],
                          dev_share_grad,
                          cacheflag,
                          cachemap,
                          destination_vertex,
                          vertex_size,
                          graph_,
                          cudaStream);
      }

      //todo: 消除 CPU 梯度
      output.top().backward(output_grad[top_idx()], retain_graph);

      pop_one_op();
    //LOG_INFO("FINISH NN OP");
    } else {
      LOG_INFO("NOT SUPPORT OP");
      assert(true);
    }
  }
    reset();  
  }
    void self_backward_test(bool retain_graph = true){
    pop_one_op();
      while (count > 1 || (count == 1 && NNOP == op.top())) {
    if (GRAPHOP == op.top()) {
      pop_one_op();
    }else if (SELFNNOP == op.top()) { 
      pop_one_op();
    } else if (NNOP == op.top()) {
      pop_one_op();
    } else {
      LOG_INFO("NOT SUPPORT OP");
      assert(true);
    }
  }
    reset();  
  }
  void train() {
    this->training = true;
  }

  void eval() {
    this->training = false;
  }

  bool is_train() {
    return this->training;
  }
  
  void debug(){
    printf("ADDEBUG input.size()%d\n", input.size());
    // for(int i=0;i<count;i++){
    int i=0;
     for(int k=0;k<iot_id.size();k++){
        LOG_INFO("IOT %ld %ld",iot_id[k].i_id1,iot_id[k].o_id);
    }
    while (!input.empty()) {
        if(i==0){
          LOG_INFO("input dim %d %d\t output dim %d \t OP type %d", input.top().size(0),input.top().size(1),output.top().dim(),op.top());
        }else{
          LOG_INFO("input dim %d %d \t output dim %d %d\t OP type %d", input.top().size(0),
                  input.top().size(1),output.top().size(0),output.top().size(1),op.top());  
        }
        input.pop();
        output.pop();
        op.pop();
        ntsOp.pop();
        i++;
    }
    this->output_grad.clear();
    count=0;
  }
  
  
  int top_idx(){
    return count - 1;
  }

  NtsVar cache_grad(){
    return output_grad[cachecount];
  }
//private:
  std::stack<OpType> op;
  std::stack<NtsVar> output;
  std::stack<NtsVar> input;
  std::stack<ntsOperator> ntsOp;
  std::vector<NtsVar> output_grad;
  std::vector<IOTensorId> iot_id;
  int count;
  bool training; 
  int cachecount;
//  GraphOperation *gt;
//  std::vector<CSC_segment_pinned *> subgraphs;
//  bool bi_direction;
};

} // namespace autodiff
} // namespace nts

#endif
