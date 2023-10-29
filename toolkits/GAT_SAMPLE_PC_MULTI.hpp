//
// Created by toao on 23-9-17.
//

#ifndef GNNMINI_GAT_SAMPLE_PC_MULTI_HPP
#define GNNMINI_GAT_SAMPLE_PC_MULTI_HPP
#include "core/neutronstar.hpp"
class GAT_SAMPLE_PC_MULTI_impl {
public:
    int iterations;
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
    // graph with no edge data
    Graph<Empty> *graph;
    FullyRepGraph* fully_rep_graph;
    //std::vector<CSC_segment_pinned *> subgraphs;
    // NN
    GNNDatum *gnndatum;
    NtsVar L_GT_C;
    NtsVar L_GT_G;
    NtsVar MASK;
    std::vector<NtsVar> multi_MASK_gpu;
//    NtsVar MASK_gpu;
    //GraphOperation *gt;
    PartitionedGraph *partitioned_graph;
    // Variables
    std::vector<std::vector<Parameter*>> multi_P;
//    std::vector<Parameter *> P;
    std::vector<std::vector<NtsVar>> device_X;
//    std::vector<NtsVar> X;
    std::vector<nts::ctx::NtsContext*> device_ctx;
//    nts::ctx::NtsContext* ctx;
    std::vector<VertexId> train_nids, val_nids, test_nids;

    NtsVar F;
//    NtsVar loss;
    NtsVar tt;
//    int batch;
//    long correct;
    int pipeline_num;
    std::vector<VertexId> batch_cache_num;
    VertexId top_cache_num;
    VertexId super_batchsize;
    VertexId epoch_super_batch_num;
    VertexId last_super_batch_num;
    std::vector<std::vector<VertexId>> device_cache_offset;
//    std::mutex sample_mutex;
//    std::mutex transfer_mutex;
//    std::mutex train_mutex;
    float cache_rate;
    std::vector<VertexId> cache_ids;
    int gpu_round = 0;
    std::vector<Cuda_Stream*> device_cuda_stream;
//    Cuda_Stream* cuda_stream;
    double gpu_wait_time = 0.0;
    std::vector<std::vector<at::cuda::CUDAStream>> device_torch_stream;
//    std::vector<at::cuda::CUDAStream> torch_stream;
    tbb::concurrent_queue<NtsVar> shared_W0_queue;
    tbb::concurrent_queue<NtsVar> shared_W1_queue;
    NCCL_Communicator* train_nccl_communicator;
    NCCL_Communicator* transfer_nccl_communicator;
    std::vector<unsigned long> corrects;
    std::vector<unsigned int> multi_batchsize;

    std::condition_variable cache_set_cv;
    std::mutex cache_set_mutex;
    std::vector<int> cache_set_batch;
    std::vector<FastSampler*> device_train_sampler;
    int num_devices;

    double exec_time = 0;
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
    double sample_time = 0.0;
    double transfer_time = 0.0;
    double train_time = 0.0;
    double update_cache_time = 0.0;
    double cpu_wait_time = 0.0;
    double cpu_mm_time = 0.0;


    void initMultiDeviceVar() {

        cudaGetDeviceCount(&num_devices);
//        num_devices = 1;
        LOG_INFO("GPU 数量: %d", num_devices);
        assert(num_devices >= 1);
        device_ctx.resize(num_devices);
        multi_MASK_gpu.resize(num_devices);
        multi_P.resize(num_devices);
        device_cuda_stream.resize(num_devices);
        device_torch_stream.resize(num_devices);
        device_train_sampler.resize(num_devices);
        device_X.resize(num_devices);
        std::vector<int> arr(num_devices);
        std::iota(arr.begin(), arr.end(), 0);
        train_nccl_communicator = new NCCL_Communicator(num_devices, arr.data());
        transfer_nccl_communicator = new NCCL_Communicator(num_devices, arr.data(), 0);
//        NCCL_Communicator::initAllNCCLComm(num_devices, arr.data());
//        for(int i = 0; i < num_devices; i++){
//            nccl_communicators.push_back(new NCCL_Communicator(i, 0));
//        }
        corrects.resize(num_devices);
        multi_batchsize.resize(num_devices);
    }


    GAT_SAMPLE_PC_MULTI_impl(Graph<Empty> *graph_, int iterations_,
                      bool process_local = false, bool process_overlap = false) {
        initMultiDeviceVar();

        graph = graph_;
        iterations = iterations_;

        active = graph->alloc_vertex_subset();
        active->fill();

        graph->init_gnnctx(graph->config->layer_string);
        graph->init_gnnctx_fanout(graph->config->fanout_string);
        // rtminfo initialize
        graph->init_rtminfo();
        graph->rtminfo->process_local = graph->config->process_local;
        graph->rtminfo->reduce_comm = graph->config->process_local;
        graph->rtminfo->copy_data = false;
        graph->rtminfo->process_overlap = graph->config->overlap;
        graph->rtminfo->with_weight = true;
        graph->rtminfo->with_cuda = true;
        graph->rtminfo->lock_free = graph->config->lock_free;
        pipeline_num = graph->config->pipeline_num;
        cache_rate = graph->config->cache_rate;
    }
    void init_graph() {

//        partitioned_graph=new PartitionedGraph(graph, active);
//        partitioned_graph->GenerateAll([&](VertexId src, VertexId dst) {
//            return nts::op::nts_norm_degree(graph,src, dst);
//        },CPU_T,true);

        fully_rep_graph = new FullyRepGraph(graph);
        fully_rep_graph->GenerateAll();
        fully_rep_graph->SyncAndLog("read_finish");
//        graph->init_communicatior();
        //cp = new nts::autodiff::ComputionPath(gt, subgraphs);
        for(int i = 0; i < num_devices; i++) {
            device_ctx[i] = new nts::ctx::NtsContext();
        }
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
        torch::manual_seed(0);
        gnndatum = new GNNDatum(graph->gnnctx, graph);
        gnndatum->init_multi_gpu(num_devices, transfer_nccl_communicator);
        // gnndatum->random_generate();
        if (0 == graph->config->feature_file.compare("random")) {
            gnndatum->random_generate();
        } else if(0 == graph->config->feature_file.compare("mask")){
            gnndatum->read_mask_random_other(graph->config->mask_file);
        }  else {
            gnndatum->readFeature_Label_Mask(graph->config->feature_file,
                                             graph->config->label_file,
                                             graph->config->mask_file);
        }

        // creating tensor to save Label and Mask
        gnndatum->registLabel(L_GT_C);
        gnndatum->registMask(MASK);
        gnndatum->generate_multi_gpu_data();
//        L_GT_G = L_GT_C.cuda();
        for(int i = 0; i < num_devices; i++) {
            multi_MASK_gpu[i] = MASK.to(torch::Device(torch::kCUDA, i));
        }

        // initializeing parameter. Creating tensor with shape [layer_size[i],
        // layer_size[i + 1]]
        for(int dev_id = 0; dev_id < num_devices; dev_id++) {
            for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
                multi_P[dev_id].push_back(new Parameter(graph->gnnctx->layer_size[i],
                                                        graph->gnnctx->layer_size[i + 1], alpha, beta1,
                                                        beta2, epsilon, weight_decay));
                multi_P[dev_id].push_back(new Parameter(graph->gnnctx->layer_size[i + 1] * 2, 1, alpha,
                                                        beta1, beta2, epsilon, weight_decay));
            }

        }

        // synchronize parameter with other processes
        // because we need to guarantee all of workers are using the same model
        for(int i = 0; i < num_devices; i++) {
            torch::Device GPU(torch::kCUDA, i);
            for (int j = 0; j < multi_P[i].size(); j++) {
                multi_P[i][j]->set_multi_gpu_comm(train_nccl_communicator);
                if (i != 0) {
                    auto &W = multi_P[0][j]->W;
                    multi_P[i][j]->W.set_data(W);
                }
                multi_P[i][j]->set_decay(decay_rate, decay_epoch);
                multi_P[i][j]->to(GPU);
                multi_P[i][j]->Adam_to_GPU(i);
            }
        }

        F = graph->Nts->NewLeafTensor(
            gnndatum->local_feature,
            {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
            torch::DeviceType::CPU);

        NtsVar d;
        for(int device_id = 0; device_id < num_devices; device_id++) {
            for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
                NtsVar d;
                device_X[device_id].push_back(d);
            }
        }        // X[0] is the initial vertex representation. We created it from
        // local_feature
        for(int device_id = 0; device_id < num_devices; device_id++) {
            device_X[device_id][0] = F.set_requires_grad(true);
        }
    }

    void Test(long s, int device_id) { // 0 train, //1 eval //2 test
        NtsVar mask_train = multi_MASK_gpu[device_id].eq(s);
        NtsVar all_train =
                device_X[device_id][graph->gnnctx->layer_size.size() - 1]
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
            if (s == 0) {
                LOG_INFO("Train Acc: %f %d %d", acc_train, g_train, g_correct);
            } else if (s == 1) {
                LOG_INFO("Eval Acc: %f %d %d", acc_train, g_train, g_correct);
            } else if (s == 2) {
                LOG_INFO("Test Acc: %f %d %d", acc_train, g_train, g_correct);
            }
        }
    }
    void Loss(int device_id) {
        //  return torch::nll_loss(a,L_GT_C);
        torch::Tensor a = device_X[device_id][graph->gnnctx->layer_size.size() - 1].log_softmax(1);
        torch::Tensor mask_train = multi_MASK_gpu[device_id].eq(0);
        auto loss = torch::nll_loss(
                a.masked_select(mask_train.expand({mask_train.size(0), a.size(1)}))
                        .view({-1, a.size(1)}),
                L_GT_G.masked_select(mask_train.view({mask_train.size(0)})));
        device_ctx[device_id]->appendNNOp(device_X[device_id][graph->gnnctx->layer_size.size() - 1], loss);
    }

    void Update(int device_id, cudaStream_t cudaStream) {
        for (int i = 0; i < multi_P[device_id].size(); i++) {
//            std::printf("layer %d W gradient sum: %.4lf\n", i, P[i]->W.grad().abs().sum().item<double>());
            // accumulate the gradient using all_reduce
            multi_P[device_id][i]->reduce_multi_gpu_gradient(multi_P[device_id][i]->W.grad(),
                                                             device_id, cudaStream);
//            multi_P[device_id][i]->all_reduce_to_gradient(multi_P[device_id][i]->W.grad().cpu());
            // update parameters with Adam optimizer
//            multi_P[device_id][i]->learnC2G_with_decay_Adam();
            multi_P[device_id][i]->learn_local_with_decay_Adam();
            multi_P[device_id][i]->next();
        }
    }

    void Train(FastSampler* sampler, int epoch_num, int device_id) {
//        LOG_INFO("device %d sampler: %p", device_id, sampler);
        graph->rtminfo->forward = true;
        corrects[device_id] = 0;
        long batch = 0;

        std::mutex sample_mutex;
        std::mutex transfer_mutex;
        std::mutex train_mutex;

        SampledSubgraph *sg[pipeline_num];
        NtsVar tmp_X0[pipeline_num];
        NtsVar tmp_target_lab[pipeline_num];

        for(int i = 0; i < pipeline_num; i++) {
            tmp_X0[i] = graph->Nts->NewLeafTensor({graph->config->batch_size,F.size(1)},
                                                  torch::DeviceType::CUDA, device_id);
            tmp_target_lab[i] = graph->Nts->NewLabelTensor({graph->config->batch_size},
                                                           torch::DeviceType::CUDA, device_id);
        }

        std::thread threads[pipeline_num];
        std::vector<unsigned int> super_batch_countdown(epoch_super_batch_num);
        std::vector<unsigned int> super_batch_ready(epoch_super_batch_num);

        for(int tid = 0; tid < pipeline_num; tid++) {
            threads[tid] = std::thread([&](int thread_id) {
//                LOG_INFO("device %d thread id 0x%lx start", device_id, std::this_thread::get_id());

                cudaSetUsingDevice(device_id);
                int local_batch;
                int super_batch_id;
                MultiCacheVars* cacheVars;
                NNVars* nnVars;

                std::unique_lock<std::mutex> sample_lock(sample_mutex, std::defer_lock);
                sample_lock.lock();
                while(sampler->sample_not_finished()) {
                    // 还要确定它是这个super batch里面的第几个batch，如果是第一个batch，则负责初始化工作，如果是最后一个batch，则负责收尾工作
                    local_batch = batch;
                    super_batch_id = local_batch / pipeline_num;
                    batch++;
                    // 如果是super batch的第一个batch
                    if(local_batch % pipeline_num == 0) {
                        // 初始化cache
                        cacheVars = gnndatum->multi_new_cache_var(super_batch_id, epoch_num);
                        cudaSetUsingDevice(device_id);
                        if(cache_set_batch[super_batch_id] != epoch_num) {
                            std::unique_lock<std::mutex> cache_set_lock(cache_set_mutex, std::defer_lock);
                            cache_set_lock.lock();
//                            std::printf("device id %d thread %d debug 1.3\n", device_id, thread_id);

                            if(cache_set_batch[super_batch_id] != epoch_num) {
                                gnndatum->set_cache_index(cacheVars->cache_map, cacheVars->cache_location,
                                                          super_batch_id, batch_cache_num, cache_ids);
                                cache_set_batch[super_batch_id] = epoch_num;
                            }
//                            std::printf("device id %d thread %d debug 1.4\n", device_id, thread_id);

                            cache_set_lock.unlock();
                        }
                        super_batch_countdown[super_batch_id] = 0;
                        super_batch_ready[super_batch_id] = false;
                    } else {
                        cacheVars = gnndatum->get_multi_cache_var(super_batch_id);
                    }
                    write_add(&sample_time, -get_time());
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
//                    LOG_INFO("thread 0x%lx batch size: %d", std::this_thread::get_id(), multi_batchsize[device_id]);
                    sg[thread_id] = sampler->sample_gpu_fast_omit(multi_batchsize[device_id], thread_id,
                                                                  cacheVars->multi_dev_cache_map[device_id], super_batch_id,
                                                                  WeightType::None);
                    gpu_round++;
                    assert(sg[thread_id]->cs->stream == device_cuda_stream[device_id][thread_id].stream);
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    write_add(&sample_time, get_time());
                    sample_lock.unlock();

                    std::unique_lock<std::mutex> transfer_lock(transfer_mutex, std::defer_lock);
                    transfer_lock.lock();
                    write_add(&transfer_time, -get_time());
                    sampler->load_label_gpu(&device_cuda_stream[device_id][thread_id], sg[thread_id],
                                            tmp_target_lab[thread_id],gnndatum->dev_local_label_multi[device_id]);
                    // load feature of cacheflag -1 0, 不需要cacheflag，在sample step已经包含cacheflag信息
                    sampler->load_feature_gpu(&device_cuda_stream[device_id][thread_id], sg[thread_id],
                                              tmp_X0[thread_id],gnndatum->dev_local_feature_multi[device_id]);
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    write_add(&transfer_time, get_time());

                    super_batch_countdown[super_batch_id]++;
                    if(super_batch_id != epoch_super_batch_num - 1) {
                        if(super_batch_countdown[super_batch_id] == pipeline_num) {
                            super_batch_ready[super_batch_id] = true;
                        }
                    } else if(super_batch_countdown[super_batch_id] == last_super_batch_num) {
                        super_batch_ready[super_batch_id] = true;
                    }
                    transfer_lock.unlock();

                    while(!super_batch_ready[super_batch_id]) {
                        std::this_thread::yield();
                    }

                    std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
                    train_lock.lock();
                    write_add(&train_time, -get_time());


                    // Note：在下面进行提取
                    // 如果是进行训练的第一个batch，就进行提取相关数据
                    write_add(&gpu_wait_time, -get_time());
                    if(super_batch_countdown[super_batch_id] == pipeline_num  || (super_batch_id == epoch_super_batch_num - 1&&
                                                                                  super_batch_countdown[super_batch_id] == last_super_batch_num )) {
                        nnVars = gnndatum->multi_new_nn_var(super_batch_id, device_id);
//                        gnndatum->move_nn_data_to_gpu(nnVars, cuda_stream[thread_id].stream);
                    } else {
                        nnVars = gnndatum->get_multi_nn_var(super_batch_id, device_id);
                    }
                    write_add(&gpu_wait_time, get_time());

                    at::cuda::setCurrentCUDAStream(device_torch_stream[device_id][thread_id]);
                    Forward(sampler, tmp_X0[thread_id], sg[thread_id], &device_cuda_stream[device_id][thread_id], super_batch_id,
                            cacheVars, nnVars, device_id);
                    if(device_ctx[device_id]->is_train()) {
                        Loss(device_X[device_id][graph->gnnctx->layer_size.size()-1],
                             tmp_target_lab[thread_id], device_ctx[device_id]);
                        BackwardAndUpdate(device_id, device_cuda_stream[device_id][thread_id].stream);
                    }
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    corrects[device_id] += getCorrect(device_X[device_id][graph->gnnctx->layer_size.size()-1], tmp_target_lab[thread_id]);
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    write_add(&train_time, get_time());

                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    if((super_batch_id != epoch_super_batch_num - 1 && super_batch_countdown[super_batch_id] == pipeline_num)
                       || (super_batch_id == epoch_super_batch_num - 1 && super_batch_countdown[super_batch_id] == last_super_batch_num)) {
                        shared_W0_queue.push(multi_P[device_id][0]->W.cpu());
                        shared_W1_queue.push(multi_P[device_id][1]->W.cpu());
                    }
//                    shared_W_queue.push(P[0]->W.cpu());
                    super_batch_countdown[super_batch_id]--;
                    if(super_batch_countdown[super_batch_id] == 0) {
                        // 最后一个batch了，回收相应的空间
                        gnndatum->recycle_multi_gpu_memory(cacheVars, nnVars, super_batch_id, device_id);
                    }
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();

                    train_lock.unlock();
//                    std::printf("device %d thread id 0x%lx finish training\n", device_id, std::this_thread::get_id());

                    sample_lock.lock();
                }
                sample_lock.unlock();
//                LOG_INFO("device %d thread id 0x%lx end", device_id, std::this_thread::get_id());

            }, tid);
        }

        // 多流等待
        for(int i = 0; i < pipeline_num; i++) {
            threads[i].join();
        }
        sampler->restart();
    }

    inline void Forward(FastSampler* sampler, NtsVar& tmp_X0, SampledSubgraph* sg, Cuda_Stream* cudaStream,
                        VertexId super_batch_id, MultiCacheVars* cacheVars, NNVars* nnVars, int device_id){

        graph->rtminfo->forward = true;

        for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
            graph->rtminfo->curr_layer = i;

//            for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
//                P.push_back(new Parameter(graph->gnnctx->layer_size[i],
//                                          graph->gnnctx->layer_size[i + 1], alpha, beta1,
//                                          beta2, epsilon, weight_decay));
//                P.push_back(new Parameter(graph->gnnctx->layer_size[i + 1] * 2, 1, alpha,
//                                          beta1, beta2, epsilon, weight_decay));
//            }
            int hop = (graph->gnnctx->layer_size.size()-2) - i;
            NtsVar  X_trans;
            if(i == 0) {
                // 计算W*h，乘的是上面的第一个，相当于GCN的vertexForward
                 X_trans=device_ctx[device_id]->runVertexForward([&](NtsVar x_i_){
                                                          int layer = i;
                                                          return multi_P[device_id][2 * layer]->forward(x_i_);
                                                      },
                                                      tmp_X0);
            } else {
                // 计算W*h，乘的是上面的第一个，相当于GCN的vertexForward
                X_trans=device_ctx[device_id]->runVertexForward([&](NtsVar x_i_){
                                                  int layer = i;
                                                  return multi_P[device_id][2 * layer]->forward(x_i_);
                                              },
                                              device_X[device_id][i]);
            }

            // TODO: 下面这一步在单机中可以去掉
//            NtsVar mirror= ctx->runGraphOp<nts::op::DistGPUGetDepNbrOp>(partitioned_graph,active,X_trans);
            // TODO： 下面这两步都不涉及到通信，改起来比较容易
//            NtsVar edge_src= ctx->runGraphOp<nts::op::BatchGPUScatterSrc>(sg, graph, hop,X_trans, cudaStream);
//            NtsVar edge_dst= ctx->runGraphOp<nts::op::BatchGPUScatterDst>(sg, graph, hop,X_trans, cudaStream);
//            NtsVar e_msg=torch::cat({edge_src,edge_dst},1);
            NtsVar e_msg = device_ctx[device_id]->runGraphOp<nts::op::BatchGPUSrcDstScatterOp>(sg, graph, hop, X_trans,
                                                                             cudaStream, device_id);
            // Note：下面这个函数也不涉及通信
            NtsVar m=device_ctx[device_id]->runEdgeForward([&](NtsVar e_msg_){
                                             int layer = i;
                                             return torch::leaky_relu(multi_P[device_id][2 * layer + 1]->forward(e_msg_),0.2);
                                         },
                                         e_msg);//edge NN
            //  partitioned_graph->SyncAndLog("e_msg_in");
            // Note: 下面这个函数也不涉及通信
            NtsVar a=device_ctx[device_id]->runGraphOp<nts::op::BatchGPUEdgeSoftMax>(sg, graph, hop,m,
                                                                                     cudaStream, device_id);// edge NN
            // Note: 下面这个函数不涉及通信
            NtsVar e_msg_out=device_ctx[device_id]->runEdgeForward([&](NtsVar a_){
//                                                     return edge_src*a_;
                                                     return e_msg.slice(1, 0, e_msg.size(1)/2, 1)*a;

                                                 },
                                                 a);//Edge NN
            //            partitioned_graph->SyncAndLog("e_msg_out");
            // Note: 该函数也不涉及通信
            NtsVar nbr= device_ctx[device_id]->runGraphOp<nts::op::BatchGPUAggregateDst>(sg, graph, hop,
                                                                                         e_msg_out, cudaStream, device_id);
            // Note: 下面的函数也不涉及通信
            device_X[device_id][i+1]=device_ctx[device_id]->runVertexForward([&](NtsVar nbr_){
                return torch::relu(nbr_);
            },nbr);
            if(i == 0) {
                write_add(&update_cache_time, -get_time());
//                update_cache_time -= get_time();
                sampler->load_share_embedding(cudaStream, sg, nnVars->dev_shared_embedding,
                                              device_X[device_id][i+1], cacheVars->multi_dev_cache_map[device_id],
                                              cacheVars->multi_dev_cache_location[device_id],
                                              super_batch_id);
                write_add(&update_cache_time, get_time());
//                update_cache_time += get_time();

            }
//            partitioned_graph->SyncAndLog("hello 2");
        }

    }

    void TestMiniBatchAll(long type, unsigned long total_num) { // 0 train, //1 eval //2 test
        unsigned long correct = 0;
        for(auto&& num: corrects) {
            correct += num;
        }
        float acc = 1.0 * correct / total_num;
        if (type == 0) {
            LOG_INFO("Train Acc: %f %d %d", acc, correct, total_num);
        } else if (type == 1) {
            LOG_INFO("Eval Acc: %f %d %d", acc, correct, total_num);
        } else if (type == 2) {
            LOG_INFO("Test Acc: %f %d %d", acc, correct, total_num);
        }
    }

    void Loss(NtsVar &left,NtsVar &right,  nts::ctx::NtsContext* ctx) {
        //  return torch::nll_loss(a,L_GT_C);
        torch::Tensor a = left.log_softmax(1);
        //torch::Tensor mask_train = MASK_gpu.eq(0);
        auto loss = torch::nll_loss(a,right);
        if (ctx->training == true) {
            ctx->appendNNOp(left, loss);
        }
    }

    inline void BackwardAndUpdate(int device_id, cudaStream_t cudaStream){
        if(device_ctx[device_id]->training) {
            device_ctx[device_id]->self_backward(false);
            Update(device_id, cudaStream);

            for(int i = 0; i < multi_P[device_id].size(); i++) {
                multi_P[device_id][i]->zero_grad();
            }
        }
    }

    long getCorrect(NtsVar &input, NtsVar &target) {
        // NtsVar predict = input.log_softmax(1).argmax(1);
        NtsVar predict = input.argmax(1);
        NtsVar output = predict.to(torch::kLong).eq(target).to(torch::kLong);
        return output.sum(0).item<long>();
    }

    void run() {
        if (graph->partition_id == 0) {
            LOG_INFO("GNNmini::[Dist.GPU.GATimpl] running [%d] Epoches\n",
                     iterations);
        }

        std::printf("pipeline num: %d\n", pipeline_num);

        for (VertexId i = 0; i < graph->gnnctx->l_v_num; ++i) {
            int type = gnndatum->local_mask[i];
            if (type == 0) {
                train_nids.push_back(i);
            } else if (type == 1) {
                val_nids.push_back(i);
            } else if (type == 2) {
                test_nids.push_back(i);
            }
        }

//        nts::op::shuffle_vec(train_nids);
//        nts::op::shuffle_vec(val_nids);
//        nts::op::shuffle_vec(test_nids);

//        nts::op::nts_local_shuffle(train_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);
//        nts::op::nts_local_shuffle(val_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);
//        nts::op::nts_local_shuffle(test_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);


        std::vector<int> fanout(1);
        fanout[0] = graph->gnnctx->fanout[graph->gnnctx->fanout.size() - 1];
        initCacheVariable();
        std::printf("cache ids size: %d\n", cache_ids.size());
        nts::op::nts_local_shuffle(train_nids, graph->config->batch_size * pipeline_num, cache_ids, batch_cache_num);


        InitStream();

        initMultiSampler();

        // TODO: 下面的cache_ids为空了
        FastSampler* cpu_sampler = new FastSampler(graph, fully_rep_graph,
                                                   cache_ids, 1, fanout,
                                                   super_batchsize);
        cpu_sampler->set_cpu_merge_src_dst();

        auto total_super_batch_num = epoch_super_batch_num * iterations;
        // 初始化cache map这些变量
        gnndatum->init_multi_cache_var(epoch_super_batch_num, num_devices);
        // 首先将一个W放进相应队列里面
        shared_W0_queue.push(multi_P[0][0]->W.cpu());
        shared_W1_queue.push(multi_P[0][1]->W.cpu());
        // 使用一个多线程queue来进行存储CPU的计算结果，然后GPU会从这里面提取计算结果

        // 使用一个特定的流来进行传输
        // 为每个GPU都创建一个流
        cudaStream_t cpu_cuda_streams[num_devices];
        for(int i = 0; i < num_devices; i++) {
            cudaSetUsingDevice(i);
            cudaStreamCreateWithFlags(&cpu_cuda_streams[i], cudaStreamNonBlocking);
        }

        auto* CPU_sg = cpu_sampler->sample_fast(batch_cache_num[0], WeightType::None);
        auto embedding = CPU_Forward(CPU_sg);

        gnndatum->move_embedding_to_multi_gpu(top_cache_num, embedding.size(0), embedding.size(1),
                                              embedding.accessor<ValueType, 2>().data(),
                                              cpu_cuda_streams, num_devices, device_cache_offset[0]);

        auto cpu_thread = std::thread([&](){
           auto max_threads = std::thread::hardware_concurrency();
           for(VertexId i = 1; i < total_super_batch_num; i++) {
               if(i % epoch_super_batch_num == 0) {
                   cpu_sampler->restart();
               }
               CPU_sg = cpu_sampler->sample_fast(batch_cache_num[i % epoch_super_batch_num], WeightType::None);

               embedding = CPU_Forward(CPU_sg);
               gnndatum->move_embedding_to_multi_gpu(top_cache_num, embedding.size(0), embedding.size(1),
                                                    embedding.accessor<ValueType, 2>().data(),
                                                    cpu_cuda_streams, num_devices,
                                                    device_cache_offset[i % epoch_super_batch_num]);

           }
        });


        auto total_node_num = train_nids.size();
        exec_time -= get_time();
        auto run_time = -get_time();
        for (int i_i = 0; i_i < iterations; i_i++) {
            double per_epoch_time = -get_time();
            graph->rtminfo->epoch = i_i;
            if (i_i != 0) {
                for(int device_id = 0; device_id < num_devices; device_id++) {
                    for (int i = 0; i < multi_P[device_id].size(); i++) {
                        multi_P[device_id][i]->zero_grad();
                    }

                }
            }
            for(int device_id = 0; device_id < num_devices; device_id++) {
                device_ctx[device_id]->train();
            }
            std::vector<std::thread> multi_thread;
            for(int device_id = 0; device_id < num_devices; device_id++) {
                multi_thread.emplace_back([&](int dev_id){
                    Train(device_train_sampler[dev_id], i_i,  dev_id);
                }, device_id);
            }
            for(int device_id = 0; device_id < num_devices; device_id++) {
                multi_thread[device_id].join();
            }
            TestMiniBatchAll(0, total_node_num);
//            Forward();
//
//
//            //      printf("sizeof %d",sizeof(__m256i));
////      printf("sizeof %d",sizeof(int));
//            Test(0);
//            Test(1);
//            Test(2);
//            Loss();
//
//            ctx->self_backward(true);
//            Update();
            per_epoch_time += get_time();
            // ctx->debug();
            std::cout << "GNNmini::Running.Epoch[" << i_i << "]:Times["
                      << per_epoch_time << "(s)]" << std::endl;
        }
        run_time += get_time();
        cpu_thread.join();
        std::printf("#run time: %.4lf (s)\n", run_time);
        std::printf("#sample time: %.4lf (s)\n", sample_time);
        std::printf("#transfer time: %.4lf (s)\n", transfer_time);
        std::printf("#train time: %.4lf (s)\n", train_time);
        std::printf("#gpu wait time: %.4lf (s)\n", gpu_wait_time);
        std::printf("#cpu wait time: %.4lf (s)\n", cpu_wait_time);
        std::printf("#cpu mm time: %.4lf (s)\n", cpu_mm_time);


        delete active;
    }

    void initCacheVariable() {

        // 进行预采样
        VertexId sample_neighs = 1;
        for(int i = 0; i < graph->gnnctx->fanout.size() - 1; i++) {
            sample_neighs *= graph->gnnctx->fanout[i];
        }
        std::printf("sample last layer neighs: %u\n", sample_neighs);
//        batch_cache_num = graph->config->batch_size * pipeline_num * cache_rate * sample_neighs;
//        std::printf("batch cache num: %u\n", batch_cache_num);
        super_batchsize = graph->config->batch_size * pipeline_num;

        auto pre_sample_time = -get_time();
        std::printf("train nids size: %d\n", train_nids.size());
        cache_ids = nts::op::preSample(train_nids, graph->config->batch_size, batch_cache_num, cache_rate, top_cache_num,
                                       gnndatum->gnnctx->layer_size.size() - 1, fully_rep_graph,
                                       cache_rate / 0.3, graph,  pipeline_num);
        pre_sample_time += get_time();
        std::printf("预采样时间: %.4lf\n", pre_sample_time);


        // 计算CPU循环次数
        // CPU计算的次数可由epoch的数量和一个epoch内super batch的数量确定
        epoch_super_batch_num = train_nids.size() / super_batchsize;
        last_super_batch_num = train_nids.size() - super_batchsize * epoch_super_batch_num;
        last_super_batch_num = std::ceil((double)last_super_batch_num/(double)graph->config->batch_size);
        if(last_super_batch_num != 0) {
            epoch_super_batch_num++;
        } else {
            last_super_batch_num = pipeline_num;
        }

        // 为每个GPU初始化传输偏移量
        device_cache_offset.resize(epoch_super_batch_num);
        for(int i = 0; i < epoch_super_batch_num; i++) {
            device_cache_offset[i].resize(num_devices+1);
            size_t per_device = batch_cache_num[i] / num_devices;
            for(int j = 0; j < num_devices; j++) {
                device_cache_offset[i][j] = per_device * i;
            }
            device_cache_offset[i][num_devices] = batch_cache_num[i];
        }
//        device_cache_offset.resize(num_devices+1);
//        size_t per_device = batch_cache_num / num_devices;
//        for(int i = 0; i < num_devices; i++) {
//            device_cache_offset[i] = per_device * i;
//        }
//
        // TODO: 下面这行肯定错的，因为每回只传输一个superbatch的大小，不是整个cache数量
//        device_cache_offset[num_devices] = cache_ids.size();
        cache_set_batch.resize(epoch_super_batch_num);
        memset(cache_set_batch.data(), -1, cache_set_batch.size() * sizeof(int));

    }

    // 这个函数应该在shuffle之后调用
    inline void initMultiSampler(){

        unsigned int per_batchsize = graph->config->batch_size / num_devices;
        unsigned int last_batchsize = graph->config->batch_size - num_devices * per_batchsize;
        last_batchsize += per_batchsize;
        int layer = graph->gnnctx->layer_size.size()-1;
        for(int i = 0; i < num_devices - 1; i++) {
            multi_batchsize[i] = per_batchsize;
        }
        multi_batchsize[num_devices - 1] = last_batchsize;
        // 确定每个batch的顶点id
        std::vector<std::vector<VertexId>> multi_train_nids(num_devices);
        // 确定batch数量
        uint32_t total_batchsize = graph->config->batch_size;
        uint32_t total_batch_num = train_nids.size() / total_batchsize;
        if(train_nids.size() % total_batchsize != 0) {
            total_batch_num++;
        }

        // 分配相应的空间
        uint32_t start = 0;
        uint32_t per_device = train_nids.size() / num_devices;
        for(int i = 0; i < num_devices; i++) {
            int num = per_device;
            if(i == num_devices - 1) {
                num = train_nids.size() - start;
            }
            multi_train_nids[i].resize(num);
            std::printf("device %d train num: %d\n", i, num);
//            std::memcpy(multi_train_nids[i].data(), train_nids.data() + start, num * sizeof(VertexId));
            start += num;
        }

        // 最后一个batch的大小
        last_batchsize = train_nids.size() % total_batchsize;
        uint32_t  last_per_device = last_batchsize / num_devices;
        uint32_t  last_per_device_end = last_batchsize % last_per_device + last_per_device;

        // 复制顶点id到相应数组
        start = 0;
        for(uint32_t i = 0; i < total_batch_num; i++) {
            // 如果是最后一个batch且相除不为零
            if(i == total_batch_num - 1 && last_batchsize != 0) {
                assert(start + last_per_device * num_devices + last_batchsize % last_per_device == train_nids.size());
                for(int j = 0; j < num_devices; j++) {
                    size_t copy_size = last_per_device * sizeof(VertexId);
                    if(j == num_devices - 1) {
                        copy_size = last_per_device_end * sizeof(VertexId);
                    }
                    assert(i * multi_batchsize[j] < multi_train_nids[j].size());
                    assert(start + last_per_device * j < train_nids.size());
                    assert(i * multi_batchsize[j] + copy_size / sizeof(VertexId) == multi_train_nids[j].size());
                    memcpy(multi_train_nids[j].data() + i * multi_batchsize[j], train_nids.data() + start + last_per_device * j,
                           copy_size);
                }
            } else {
                // 按照偏移量进行复制
                for(int j = 0; j < num_devices; j++) {
                    assert(i * multi_batchsize[j] < multi_train_nids[j].size());
                    assert(start + per_batchsize * j < train_nids.size());
                    memcpy(multi_train_nids[j].data() + i * multi_batchsize[j], train_nids.data() + start + per_batchsize * j,
                           multi_batchsize[j] * sizeof(VertexId));
                }
            }

            start += total_batchsize;
        }

        for(int i = 0; i < num_devices; i++) {
            cudaSetUsingDevice(i);
            std::printf("train id size: %d\n", multi_train_nids[i].size());
            device_train_sampler[i] = new FastSampler(fully_rep_graph,multi_train_nids[i],layer,
                                                      graph->gnnctx->fanout, pipeline_num, device_cuda_stream[i]);
            device_train_sampler[i]->set_merge_src_dst(pipeline_num);
//            for(int j = 0; j < pipeline_num; j++)
//                device_cuda_stream[i][j].CUDA_DEVICE_SYNCHRONIZE();
            std::printf("sampler train id size: %d\n", device_train_sampler[i]->work_range[1]);
        }
    }

    NtsVar CPU_Forward(SampledSubgraph* CPU_sg) {
        NtsVar W0, W1;

        cpu_wait_time -= get_time();
        while(!shared_W0_queue.try_pop(W0)) {
            std::this_thread::yield();
        }
        cpu_wait_time += get_time();
//        print_tensor_size(W0);

        cpu_mm_time -= get_time();
        NtsVar index = torch::from_blob(CPU_sg->sampled_sgs[0]->source.data(),
                                        {static_cast<long>(CPU_sg->sampled_sgs[0]->source.size())},
                                        torch::kInt32);
        NtsVar CPU_X = F.index_select(0, index);
        NtsVar X_trans = CPU_X.matmul(W0).set_requires_grad(false);
        cpu_mm_time += get_time();
//        print_tensor_size(X_trans);

        NtsVar E_msg = device_ctx[0]->runGraphOpNoBackward<nts::op::PushDownCPUSrcDstScatterOp>(CPU_sg, graph, 0, X_trans);
//        print_tensor_size(E_msg);

        cpu_wait_time -= get_time();
        while(!shared_W1_queue.try_pop(W1)) {
            std::this_thread::yield();
        }
        cpu_wait_time += get_time();
//        print_tensor_size(W1);

        NtsVar m = torch::leaky_relu(E_msg.matmul(W1), 0.2).set_requires_grad(false);
//        print_tensor_size(m);

        NtsVar a = device_ctx[0]->runGraphOpNoBackward<nts::op::PushDownEdgeSoftMax>(CPU_sg, graph, 0, m);
//        print_tensor_size(a);

        NtsVar E_msg_out = (E_msg.slice(1, 0, E_msg.size(1) / 2, 1) * a).set_requires_grad(false);
//        print_tensor_size(E_msg_out);

        NtsVar nbr = device_ctx[0]->runGraphOpNoBackward<nts::op::PushDownCPUDstAggregateOp>(CPU_sg, graph, 0, E_msg_out);
//        print_tensor_size(nbr);

        return torch::relu(nbr).set_requires_grad(false);

    }

    inline void InitStream() {
        for(int deviceId = 0; deviceId < num_devices; deviceId++) {
            cudaSetUsingDevice(deviceId);
            device_cuda_stream[deviceId] = new Cuda_Stream[pipeline_num];
            for(int i = 0; i < pipeline_num; i++) {
                cudaStream_t cudaStream;
                cudaStreamCreateWithFlags(&(cudaStream), cudaStreamNonBlocking);
                // Note: 这里是每个pipeline一个stream
                device_cuda_stream[deviceId][i].setNewStream(cudaStream);
                device_torch_stream[deviceId].emplace_back(at::cuda::CUDAStream(
                        at::cuda::CUDAStream::UNCHECKED,
                        at::Stream(
                                at::Stream::UNSAFE,
                                c10::Device(at::DeviceType::CUDA, deviceId),
                                reinterpret_cast<int64_t>(cudaStream)))
                );

//            torch_stream.push_back(at::cuda::getStreamFromPool(true));
//            auto stream = torch_stream[i].stream();
//            cuda_stream[i].setNewStream(stream);
//            for(int j = 0; j < i; j++) {
//                if(cuda_stream[j].stream == stream|| stream == default_stream) {
//                    std::printf("stream i:%p is repeat with j: %p, default: %p\n", stream, cuda_stream[j].stream, default_stream);
//                    exit(3);
//                }
//            }
            }

        }
    }

};

#endif //GNNMINI_GAT_SAMPLE_PC_MULTI_HPP
