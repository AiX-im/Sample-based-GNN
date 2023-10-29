#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include "cuda/ntsCudaTest.cuh"
class GCN_SAMPLE_GPU_impl {
public:
  int iterations;
  int pipeline_num;
  ValueType learn_rate;
  ValueType weight_decay;
  ValueType drop_rate;
  ValueType alpha;
  ValueType beta1;
  ValueType beta2;
  ValueType epsilon;
  ValueType decay_rate;
  ValueType decay_epoch;

  // graph
  VertexSubset *active;
  Graph<Empty> *graph;
  //std::vector<CSC_segment_pinned *> subgraphs;
  // NN
  GNNDatum *gnndatum;
  NtsVar L_GT_C;
  NtsVar L_GT_G;
  NtsVar MASK;
  NtsVar MASK_gpu;
  //GraphOperation *gt;
  PartitionedGraph *partitioned_graph;
  nts::ctx::NtsContext *ctx;
  // Variables
  std::vector<Parameter *> P;
  std::vector<NtsVar> X;
  Cuda_Stream* cuda_stream;
  cudaEvent_t* cuda_event;
  std::vector<at::cuda::CUDAStream> torch_stream;
//  Cuda_Stream** cuda_streams;

  NtsVar F;
  NtsVar loss;
  NtsVar tt;
  torch::nn::Dropout drpmodel;
  FullyRepGraph* fully_rep_graph;
  float acc;
  int batch;
  long correct;
  double exec_time = 0;
  double training_time = 0;
  double transfer_sample_time = 0;
  double transfer_feature_time = 0;
  double gather_feature_time = 0;
  double sample_time = 0;
  double all_sync_time = 0;
  double sync_time = 0;
  double all_graph_sync_time = 0;
  double graph_sync_time = 0;
  double all_compute_time = 0;
  double compute_time = 0;
  double all_copy_time = 0;
  double copy_time = 0;
  double graph_time = 0;
  double all_graph_time = 0;
  double* wait_times;

  std::mutex sample_mutex;
  std::mutex transfer_mutex;
  std::mutex train_mutex;

  GCN_SAMPLE_GPU_impl(Graph<Empty> *graph_, int iterations_,
                        bool process_local = false,
                        bool process_overlap = false) {
    graph = graph_;
    iterations = iterations_;

    active = graph->alloc_vertex_subset();
    active->fill();

    graph->init_gnnctx(graph->config->layer_string);
    graph->init_gnnctx_fanout(graph->config->fanout_string);
    graph->init_rtminfo();
    graph->rtminfo->set(graph->config);
    //        graph->rtminfo->process_local = graph->config->process_local;
    //        graph->rtminfo->reduce_comm = graph->config->process_local;
    //        graph->rtminfo->lock_free=graph->config->lock_free;
    //        graph->rtminfo->process_overlap = graph->config->overlap;

    graph->rtminfo->with_weight = true;
    graph->rtminfo->with_cuda = true;
    graph->rtminfo->copy_data = false;
    pipeline_num = graph->config->pipeline_num;
    if(pipeline_num <= 0) {
      pipeline_num = 3;
    }
    wait_times = new double[pipeline_num];
    std::printf("pipeline num: %d\n", pipeline_num);
  }
  void init_graph() {

    fully_rep_graph=new FullyRepGraph(graph);
    fully_rep_graph->GenerateAll();
    fully_rep_graph->SyncAndLog("read_finish");
  
    // graph->init_message_buffer();
    // graph->init_communicatior();
    ctx=new nts::ctx::NtsContext();
  }
  void init_nn() {

    learn_rate = graph->config->learn_rate;
    weight_decay = graph->config->weight_decay;
    drop_rate = graph->config->drop_rate;
    alpha = graph->config->learn_rate;
    decay_rate = graph->config->decay_rate;
    decay_epoch = graph->config->decay_epoch;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-9;

    gnndatum = new GNNDatum(graph->gnnctx, graph);
    if (0 == graph->config->feature_file.compare("random")) {
      gnndatum->random_generate();
    } else if(0 == graph->config->feature_file.compare("mask")){
        gnndatum->read_mask_random_other(graph->config->mask_file);
    }  else {
      gnndatum->readFeature_Label_Mask(graph->config->feature_file,
                                       graph->config->label_file,
                                       graph->config->mask_file);
    }
    gnndatum->registLabel(L_GT_C);
    gnndatum->registMask(MASK);
    gnndatum->genereate_gpu_data();
    // L_GT_G = L_GT_C.cuda();
    MASK_gpu = MASK.cuda();

    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i + 1], alpha, beta1, beta2, epsilon, weight_decay));
      //            P.push_back(new Parameter(graph->gnnctx->layer_size[i],
      //                        graph->gnnctx->layer_size[i+1]));
    }

    torch::Device GPU(torch::kCUDA, 0);
    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
      P[i]->to(GPU);
      P[i]->Adam_to_GPU();
    }

    drpmodel = torch::nn::Dropout(
        torch::nn::DropoutOptions().p(drop_rate).inplace(false));

    //        F=graph->Nts->NewOnesTensor({graph->gnnctx->l_v_num,
    //        graph->gnnctx->layer_size[0]},torch::DeviceType::CPU);

    F = graph->Nts->NewLeafTensor(
        gnndatum->local_feature,
        {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
        torch::DeviceType::CPU);

    for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
      NtsVar d;
      X.push_back(d);
    }
    // X[0]=F.cuda().set_requires_grad(true);
     X[0] = F.set_requires_grad(true);
  }

  long getCorrect(NtsVar &input, NtsVar &target) {
    // NtsVar predict = input.log_softmax(1).argmax(1);
    NtsVar predict = input.argmax(1);
    NtsVar output = predict.to(torch::kLong).eq(target).to(torch::kLong);
    return output.sum(0).item<long>();
  }

  void shuffle_vec(std::vector<VertexId>& vec) {
    unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count();
    std::shuffle (vec.begin(), vec.end(), std::default_random_engine(seed));
  }
  
  void Test(long s) { // 0 train, //1 eval //2 test
    NtsVar mask_train = MASK_gpu.eq(s);
    NtsVar all_train =
        X[graph->gnnctx->layer_size.size() - 1]
            .argmax(1)
            .to(torch::kLong)
            .eq(L_GT_G)
            .to(torch::kLong)
            .masked_select(mask_train.view({mask_train.size(0)}));
    NtsVar all = all_train.sum(0).cpu();
    long *p_correct = all.data_ptr<long>();
    long g_correct = 0;
    long p_train = all_train.size(0);
    long g_train = 0;
    MPI_Datatype dt = get_mpi_data_type<long>();
    MPI_Allreduce(p_correct, &g_correct, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&p_train, &g_train, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    float acc_train = 0.0;
    if (g_train > 0)
      acc_train = float(g_correct) / g_train;
    if (graph->partition_id == 0) {
      if (s == 0)
        std::cout << "Train ACC: " << acc_train << " " << g_train << " "
                  << g_correct << std::endl;
      else if (s == 1)
        std::cout << "Eval  ACC: " << acc_train << " " << g_train << " "
                  << g_correct << " " << std::endl;
      else if (s == 2)
        std::cout << "Test  ACC: " << acc_train << " " << g_train << " "
                  << g_correct << " " << std::endl;
    }
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
  void Loss(NtsVar &left,NtsVar &right) {
    //  return torch::nll_loss(a,L_GT_C);
    torch::Tensor a = left.log_softmax(1);
    //torch::Tensor mask_train = MASK_gpu.eq(0);
    loss = torch::nll_loss(a,right);
    if (ctx->training == true) {
      ctx->appendNNOp(left, loss);
    }    
  }

  void Update() {
      // auto curr_stream = at::cuda::getCurrentCUDAStream().stream();
      // std::printf("Update current stream: %p\n", curr_stream);
    for (int i = 0; i < P.size(); i++) {
      // P[i]->set_gradient(P[i]->W.grad().cuda());
      P[i]->learn_local_with_decay_Adam();
      P[i]->next();
    }
  }

  NtsVar vertexForward(NtsVar &a, NtsVar &x) {
      // auto curr_stream = at::cuda::getCurrentCUDAStream().stream();
      // std::printf("vertexForward current stream: %p\n", curr_stream);
    NtsVar y;
    int layer = graph->rtminfo->curr_layer;
    if (layer == 1) {
      y = P[layer]->forward(a);
      y = y.log_softmax(1); //CUDA

    } else if (layer == 0) {
      //y = P[layer]->forward(torch::relu(drpmodel(a)));
      y = torch::dropout(torch::relu(P[layer]->forward(a)), drop_rate, ctx->is_train());
    }
    return y;
  }


  void Forward(FastSampler* sampler, int type=0) {
      graph->rtminfo->forward = true;
      correct = 0;
      batch = 0;
//      NtsVar target_lab;
//      X[0]=graph->Nts->NewLeafTensor({1000,F.size(1)},
//        torch::DeviceType::CUDA);
//      target_lab=graph->Nts->NewLabelTensor({graph->config->batch_size},
//            torch::DeviceType::CUDA);
      SampledSubgraph *sg[pipeline_num];
      NtsVar tmp_X0[pipeline_num];
      NtsVar tmp_target_lab[pipeline_num];
      for(int i = 0; i < pipeline_num; i++) {
          tmp_X0[i] = graph->Nts->NewLeafTensor({1000,F.size(1)},
                                                torch::DeviceType::CUDA);
          tmp_target_lab[i] = graph->Nts->NewLabelTensor({graph->config->batch_size},
                                                         torch::DeviceType::CUDA);
      }
      // cudaEventRecord(cuda_event[pipeline_num - 1], cuda_stream[pipeline_num - 1].stream);
      // omp_set_nested(1);
      std::thread threads[pipeline_num];
      for(int tid = 0; tid < pipeline_num; tid++) {
        threads[tid] = std::thread([&](int thread_id){
          // std::printf("线程id: %d\n", thread_id);
          wait_times[thread_id] -= get_time();
          std::unique_lock<std::mutex> sample_lock(sample_mutex, std::defer_lock);
          sample_lock.lock();
          wait_times[thread_id] += get_time();
          while(sampler->sample_not_finished()){
              // testStream(cuda_stream[thread_id].stream);
              // std::printf("thread id: %d, before sample time: %lf\n", thread_id, sample_time);
              sample_time -= get_time();
              // std::printf("线程: %d开始采样\n", thread_id);
              sg[thread_id]=sampler->sample_fast(graph->config->batch_size, thread_id);
              // std::printf("\t线程: %d采样完成\n", thread_id);
              cudaStreamSynchronize(cuda_stream[thread_id].stream);
              sample_time += get_time();
              sample_lock.unlock();

              std::unique_lock<std::mutex> transfer_lock(train_mutex, std::defer_lock);
              transfer_lock.lock();
              transfer_feature_time -= get_time();
              //  X[0] = X[0].cuda().set_requires_grad(true);
              //  target_lab = target_lab.cuda();
//              while(thread_id != 100) {
//                  std::printf("thread_id: %d 在等待\n", thread_id);
//                  sleep(5);
//              }
            //  std::printf("\nthread_id: %d 结束等待\n", thread_id);
              sampler->load_label_gpu(&cuda_stream[thread_id], sg[thread_id], tmp_target_lab[thread_id],gnndatum->dev_local_label);

              sampler->load_feature_gpu(&cuda_stream[thread_id], sg[thread_id], tmp_X0[thread_id],gnndatum->dev_local_feature);
              cudaStreamSynchronize(cuda_stream[thread_id].stream);
              transfer_feature_time += get_time();
              transfer_lock.unlock();


              double training_time_step = 0;
              wait_times[thread_id] -= get_time();
              std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
              train_lock.lock();
              wait_times[thread_id] += get_time();
              training_time -= get_time();

              // int wait_id = thread_id - 1;
              // if(thread_id == 0) {
              //   wait_id = pipeline_num -1;
              // }
              // cudaStreamWaitEvent(cuda_stream[thread_id].stream, cuda_event[wait_id], 0);
              // std::printf("thread: %d, real thread: %d, X[0].size(0): %lu\n", thread_id, omp_get_thread_num(), X[0].size(0));

              // assert(torch_stream[thread_id].stream() == cuda_stream[thread_id].stream);
              at::cuda::setCurrentCUDAStream(torch_stream[thread_id]);
              // X[0] = tmp_X0[thread_id];
              // std::printf("default: %p, setting stream :%p\n",at::cuda::getDefaultCUDAStream().stream(), torch_stream[thread_id].stream());
              for(int l=0;l<(graph->gnnctx->layer_size.size()-1);l++){//forward
                  graph->rtminfo->curr_layer = l;
                  int hop=(graph->gnnctx->layer_size.size()-2)-l;
                  if(l == 0) {
                    NtsVar Y_i=ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(sg[thread_id],graph,hop,tmp_X0[thread_id],&cuda_stream[thread_id]);
                    X[l + 1] = ctx->runVertexForward([&](NtsVar n_i,NtsVar v_i){
                                                        // std::printf("vertexForward, size:(%d, %d)\n", v_i.size(0), v_i.size(1));
                                                       return vertexForward(n_i, v_i);
                                                   },
                                                   Y_i,
                                                   tmp_X0[thread_id]);
                  } else {

                    NtsVar Y_i=ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(sg[thread_id],graph,hop,X[l],&cuda_stream[thread_id]);
                    X[l + 1] = ctx->runVertexForward([&](NtsVar n_i,NtsVar v_i){
                                                        // std::printf("vertexForward, size:(%d, %d)\n", v_i.size(0), v_i.size(1));
                                                       return vertexForward(n_i, v_i);
                                                   },
                                                   Y_i,
                                                   X[l]);
                  }
                  
              }

              Loss(X[graph->gnnctx->layer_size.size()-1],tmp_target_lab[thread_id]);
              if (ctx->training) {
                  training_time_step -= get_time();
                  ctx->self_backward(false);
                  //print_cuda_use();
                  training_time_step += get_time();
                  //printf("#training_time_step=%lf(s)\n", training_time_step * 10);
                  Update();
                  for (int i = 0; i < P.size(); i++) {
                      P[i]->zero_grad();
                  }
              }
              cudaStreamSynchronize(cuda_stream[thread_id].stream);
              training_time += get_time();

              correct += getCorrect(X[graph->gnnctx->layer_size.size()-1], tmp_target_lab[thread_id]);
              batch++;
              // std::printf("thread %d 完成了训练\n", thread_id);
              train_lock.unlock();
              wait_times[thread_id] -= get_time();
              sample_lock.lock();
              wait_times[thread_id] += get_time();
          }
          sample_lock.unlock();
          // std::printf("采样总次数：%d\n", batch);
      }, tid);
      }
      for(int i = 0; i < pipeline_num; i++) {
        threads[i].join();
      }
       sampler->restart();
      acc = 1.0 * correct / sampler->work_range[1];
      if (type == 0) {
        LOG_INFO("Train Acc: %f %d %d", acc, correct, sampler->work_range[1]);
      } else if (type == 1) {
        LOG_INFO("Eval Acc: %f %d %d", acc, correct, sampler->work_range[1]);
      } else if (type == 2) {
        LOG_INFO("Test Acc: %f %d %d", acc, correct, sampler->work_range[1]);
      }
      //loss=X[graph->gnnctx->layer_size.size()-1];
    // #run_time=1784.571165(s)
    // exec_time=1842.431756(s) reddit
    
    // run_time=2947.184987(s) cpu
    // exec_time=2986.859283(s)
  }

  void run() {
    if (graph->partition_id == 0)
      printf("GNNmini::Engine[Dist.GPU.GCNimpl] running [%d] Epochs\n",
             iterations);

    //      graph->print_info();
    std::vector<VertexId> train_nids, val_nids, test_nids;
      for (int i = 0; i < graph->gnnctx->l_v_num; ++i) {
        int type = gnndatum->local_mask[i];
        if (type == 0) {
          train_nids.push_back(i);
        } else if (type == 1) {
          val_nids.push_back(i);
        } else if (type == 2) {
          test_nids.push_back(i);
        }
    }
      
    //NtsVar f_input_grad = graph->Nts->NewKeyTensor({232965,602},torch::DeviceType::CUDA);


    shuffle_vec(train_nids);
    shuffle_vec(val_nids);
    shuffle_vec(test_nids);
    // Sampler* train_sampler = new Sampler(fully_rep_graph, train_nids,graph->gnnctx->layer_size.size()-1,graph->config->batch_size,graph->gnnctx->fanout);
    // Sampler* eval_sampler = new Sampler(fully_rep_graph, val_nids,graph->gnnctx->layer_size.size()-1,graph->config->batch_size,graph->gnnctx->fanout);
    // Sampler* test_sampler = new Sampler(fully_rep_graph, test_nids,graph->gnnctx->layer_size.size()-1,graph->config->batch_size,graph->gnnctx->fanout);


    cuda_stream = new Cuda_Stream[pipeline_num];
    auto default_stream = at::cuda::getDefaultCUDAStream();
    for(int i = 0; i < pipeline_num; i++) {
        // cudaStreamCreateWithFlags(&(cuda_stream[i].stream), cudaStreamNonBlocking);
        // torch_stream.emplace_back(at::cuda::CUDAStream(
        //     at::cuda::CUDAStream::UNCHECKED,
        //     at::Stream(
        //         at::Stream::UNSAFE,
        //         c10::Device(at::DeviceType::CUDA, 0),
        //         reinterpret_cast<int64_t>(cuda_stream[i].stream)))
        // );


        torch_stream.push_back(at::cuda::getStreamFromPool(true));
        auto stream = torch_stream[i].stream();
        cuda_stream[i].setNewStream(stream);
        for(int j = 0; j < i; j++) {
            if(cuda_stream[j].stream == stream|| stream == default_stream) {
                std::printf("stream i:%p is repeat with j: %p, default: %p\n", stream, cuda_stream[j].stream, default_stream);
                exit(3);
            }
        }
    }
    // cuda_event = new cudaEvent_t[pipeline_num];
    // for(int i = 0; i < pipeline_num; i++) {
    //   cudaEventCreate(&cuda_event[i]);
    // }

    int layer = graph->gnnctx->layer_size.size()-1;
    if(graph->config->pushdown)
        layer--;
    FastSampler* train_sampler = new FastSampler(graph,fully_rep_graph, 
        train_nids,layer,
            graph->gnnctx->fanout,graph->config->batch_size,true, 0, pipeline_num, cuda_stream);

    FastSampler* eval_sampler = new FastSampler(graph,fully_rep_graph, 
        val_nids,layer,
            graph->gnnctx->fanout,graph->config->batch_size,true);

    FastSampler* test_sampler = new FastSampler(graph,fully_rep_graph,
        test_nids,layer,
            graph->gnnctx->fanout,graph->config->batch_size,true);

//    cuda_stream = new Cuda_Stream();

    exec_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++) {
      double per_epoch_time = 0.0;
      per_epoch_time -= get_time();
      graph->rtminfo->epoch = i_i;
      if (i_i != 0) {
        for (int i = 0; i < P.size(); i++) {
          P[i]->zero_grad();
        }
      }
      ctx->train();
      Forward(train_sampler, 0);
      // ctx->eval();
      // Forward(eval_sampler, 1);
      // Forward(test_sampler, 2);
      per_epoch_time += get_time();

      std::cout << "GNNmini::Running.Epoch[" << i_i << "]:Times["
                  << per_epoch_time << "(s)]:loss\t" << loss << std::endl;
    }

    exec_time += get_time();
    auto wait_time = 0.0;
    for(int i = 0; i < pipeline_num; i++) {
      wait_time += wait_times[i];
    }
    auto inclusive_time = 0.0;
    for(int i = 0; i < pipeline_num; i++) {
      inclusive_time += cuda_stream[i].inclusiveTime;
    }
    printf("#run_time=%lf(s)\n", exec_time);
    printf("#sample pre time: %ld(s)\n", train_sampler->pre_pro_time);
    printf("#inclusive time: %lf(s)\n", inclusive_time);
    printf("#sample post time:%lf(s)\n",train_sampler->post_pro_time);
    printf("#total sample time: %lf(s)\n", sample_time);
    printf("#wait time: %lf(s)\n", wait_time);
    printf("#sample_time=%lf(s)\n", (sample_time - train_sampler->copy_gpu_time - train_sampler->post_pro_time) );
    printf("#transfer_feature_time=%lf(s)\n", (transfer_feature_time + train_sampler->copy_gpu_time));
    printf("#training_time=%lf(s)\n", training_time);
    printf("#gather_feature_time=%lf(s)\n", gather_feature_time);
    delete active;
  }

  // void cachenode_select(){
  //     MPI_Datatype vid_t = get_mpi_data_type<VertexId>();
  //     ntsc->sample_num = graph->alloc_interleaved_vertex_array<VertexId>();
  //     for (VertexId v_i = 0; v_i < graph->vertices; v_i++) {
  //         ntsc->sample_num[v_i] = 0;
  //     }
  //     while(sampler_all->sample_not_finished()){
  //           sampler_all->reservoir_sample_all(graph->gnnctx->layer_size.size()-1,
  //                                     graph->config->batch_size,
  //                                     graph->gnnctx->fanout,
  //                                     ntsc->cacheflag);
  //     }
      
  //     SampledSubgraph *sg;
  //     while(sampler_all->has_rest()){
  //       sg=sampler_all->get_one();
  //       for(auto id:sg->sampled_sgs[graph->gnnctx->layer_size.size()-2]->dst()){
  //           ntsc->sample_num[id]++;
  //       }
  //     }
  //     MPI_Allreduce(MPI_IN_PLACE, ntsc->sample_num, graph->vertices, vid_t, MPI_SUM,
  //                 MPI_COMM_WORLD);
  //     sampler_all->useagain();
  //     for (VertexId v_i = 0; v_i < graph->vertices; v_i++) {
  //         ntsc->sample_sort[v_i] = ntsc->sample_num[v_i];
  //     }
  //     sort(ntsc->sample_sort.begin(), ntsc->sample_sort.end());
  //     ntsc->cacheSamplenum = ntsc->sample_sort[ntsc->cacheBoundary];
  //     ntsc->fanoutBoundary = ntsc->fanoutRatio * graph->gnnctx->fanout[0];

  //     for(int idx = 0; idx < graph->vertices; idx++){
  //         VertexId nbrs = fully_rep_graph->column_offset[idx+1] - fully_rep_graph->column_offset[idx];
  //         bool cacheflag = (nbrs > ntsc->fanoutBoundary) ? false : true;
  //         if(ntsc->sample_num[idx] > ntsc->cacheSamplenum && cacheflag){
  //             ntsc->partition_cache[graph->get_partition_id(idx)].push_back(idx);
  //             ntsc->cacheflag[idx] = 1;
  //             ntsc->cachenum++;
  //           }
  //     }
  //     printf("cachenum:%d,vertexs:%d",ntsc->cachenum,graph->vertices);
  //     for(int i = 0; i < ntsc->partition_cache[graph->partition_id].size();i++){
  //         VertexId idx = ntsc->partition_cache[graph->partition_id][i];
  //         ntsc->map_cache.insert(std::make_pair(idx, i));
  //     }

  //     sampler_all->clear_queue();
  //     sampler_all->restart();
  // }

};
