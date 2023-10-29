//
// Created by toao on 23-9-17.
//

#ifndef GNNMINI_GAT_SAMPLE_ALL_MULTI_HPP
#define GNNMINI_GAT_SAMPLE_ALL_MULTI_HPP
#include "core/neutronstar.hpp"
class GAT_SAMPLE_ALL_MULTI_impl {
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
//    NtsVar MASK_gpu;
    std::vector<NtsVar> multi_MASK_gpu;
    //GraphOperation *gt;
    PartitionedGraph *partitioned_graph;
    // Variables
    std::vector<std::vector<Parameter*>>  multi_P;
//    std::vector<Parameter *> P;
    std::vector<std::vector<NtsVar>> device_X;
//    std::vector<NtsVar> X;
    std::vector<nts::ctx::NtsContext *> device_ctx;
//    nts::ctx::NtsContext* ctx;

    NtsVar F;
//    NtsVar loss;
    NtsVar tt;
//    int batch;
//    long correct;
    int pipeline_num;
    VertexId batch_cache_num;
    VertexId epoch_super_batch_num;
    VertexId last_super_batch_num;
//    std::mutex sample_mutex;
//    std::mutex transfer_mutex;
//    std::mutex train_mutex;
    std::vector<VertexId> cache_ids;
    int gpu_round = 0;
    std::vector<Cuda_Stream*> device_cuda_stream;
//    Cuda_Stream* cuda_stream;
    double gpu_wait_time = 0.0;
    std::vector<std::vector<at::cuda::CUDAStream>> device_torch_stream;
//    std::vector<at::cuda::CUDAStream> torch_stream;
    tbb::concurrent_queue<NtsVar> shared_W_queue;
    std::vector<FastSampler*> device_train_sampler;
    NCCL_Communicator* train_nccl_communicator;
    std::vector<unsigned long> corrects;
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


    void initMultiDeviceVar() {

        cudaGetDeviceCount(&num_devices);
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
//        NCCL_Communicator::initAllNCCLComm(num_devices, arr.data());
//        for(int i = 0; i < num_devices; i++){
//            nccl_communicators.push_back(new NCCL_Communicator(i, 0));
//        }
        corrects.resize(num_devices);
    }

    GAT_SAMPLE_ALL_MULTI_impl(Graph<Empty> *graph_, int iterations_,
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
        gnndatum->init_multi_gpu(num_devices, train_nccl_communicator);
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
        L_GT_G = L_GT_C.cuda();
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
//        torch::Device GPU(torch::kCUDA, 0);
//        for (int i = 0; i < P.size(); i++) {
//            P[i]->init_parameter();
//            P[i]->set_decay(decay_rate, decay_epoch);
//            P[i]->to(GPU);
//            P[i]->Adam_to_GPU();
//        }
        for(int i = 0; i < num_devices; i++) {
            torch::Device GPU(torch::kCUDA, i);
            for(int j = 0; j < multi_P[i].size(); j++) {
                multi_P[i][j]->set_multi_gpu_comm(train_nccl_communicator);
                if(i != 0) {
                    auto& W = multi_P[0][j]->W;
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

    void Train(FastSampler* sampler,  int device_id) {
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


        for(int tid = 0; tid < pipeline_num; tid++) {
            threads[tid] = std::thread([&](int thread_id) {
//                LOG_INFO("device %d thread id 0x%lx start", device_id, std::this_thread::get_id());
                cudaSetUsingDevice(device_id);
                std::unique_lock<std::mutex> sample_lock(sample_mutex, std::defer_lock);
                sample_lock.lock();
                int local_batch;

                while(sampler->sample_not_finished()) {
                    write_add(&sample_time, -get_time());
                    // 如果是super batch的第一个batch
                    sg[thread_id] = sampler->sample_gpu_fast(graph->config->batch_size/num_devices, thread_id,
                                                             WeightType::None);
                    gpu_round++;
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    write_add(&sample_time, get_time());
                    sample_lock.unlock();

                    std::unique_lock<std::mutex> transfer_lock(transfer_mutex, std::defer_lock);
                    transfer_lock.lock();
                    write_add(&transfer_time, -get_time());
                    sampler->load_label_gpu(&device_cuda_stream[device_id][thread_id], sg[thread_id],
                                            tmp_target_lab[thread_id],gnndatum->dev_local_label_multi[device_id]);
                    // load feature of cacheflag -1 0, 不需要cacheflag，在sample step已经包含cacheflag信息。
                    sampler->load_feature_gpu(&device_cuda_stream[device_id][thread_id], sg[thread_id],
                                              tmp_X0[thread_id],gnndatum->dev_local_feature_multi[device_id]);
//                    sampler->load_dst_src_feature_gpu(&cuda_stream[thread_id], sg[thread_id], tmp_X0[thread_id], gnndatum->dev_local_feature);
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    write_add(&transfer_time, get_time());
                    transfer_lock.unlock();

                    std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
                    train_lock.lock();
                    write_add(&train_time, -get_time());

                    at::cuda::setCurrentCUDAStream(device_torch_stream[device_id][thread_id]);
                    Forward(sampler, tmp_X0[thread_id], sg[thread_id], &device_cuda_stream[device_id][thread_id], device_id);
                    if(device_ctx[device_id]->training) {
                        Loss(device_X[device_id][graph->gnnctx->layer_size.size()-1],
                             tmp_target_lab[thread_id], device_ctx[device_id]);
                        BackwardAndUpdate(device_id, device_cuda_stream[device_id][thread_id].stream);
                    }
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
//                    std::printf("device %d thread id 0x%lx before correct\n", device_id, std::this_thread::get_id());
                    corrects[device_id] += getCorrect(device_X[device_id][graph->gnnctx->layer_size.size()-1],
                                                      tmp_target_lab[thread_id]);
//                    std::printf("device %d thread id 0x%lx after correct\n", device_id, std::this_thread::get_id());
                    batch++;
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    write_add(&train_time, get_time());
                    train_lock.unlock();

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
    inline void Forward(FastSampler* sampler, NtsVar& tmp_X0,
                        SampledSubgraph* sg, Cuda_Stream* cudaStream, int device_id){

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
                X_trans = device_ctx[device_id]->runVertexForward([&](NtsVar x_i_){
//                                                  int layer = graph->rtminfo->curr_layer;
                                                    int layer = i;
                                                  return multi_P[device_id][2 * layer]->forward(x_i_);
                                              },
                                              tmp_X0);
            } else {
                // 计算W*h，乘的是上面的第一个，相当于GCN的vertexForward
                X_trans = device_ctx[device_id]->runVertexForward([&](NtsVar x_i_){
                                                  int layer = i;
                                                  return multi_P[device_id][2 * layer]->forward(x_i_);
                                              },
                                              device_X[device_id][i]);
            }
//            std::printf("X trans sum: %.4lf\n", X_trans.abs().sum().item<double>());
            cudaStream->CUDA_DEVICE_SYNCHRONIZE();
            // TODO: 下面这一步在单机中可以去掉
//            NtsVar mirror= ctx->runGraphOp<nts::op::DistGPUGetDepNbrOp>(partitioned_graph,active,X_trans);
            // TODO： 下面这两步都不涉及到通信，改起来比较容易
//            print_tensor_size(X_trans);
//            NtsVar edge_src= ctx->runGraphOp<nts::op::BatchGPUScatterSrc>(sg, graph, hop,X_trans,
//                                                                          cudaStream);
//            cudaStream->CUDA_DEVICE_SYNCHRONIZE();
////            LOG_DEBUG("1");
//            NtsVar edge_dst= ctx->runGraphOp<nts::op::BatchGPUScatterDst>(sg, graph, hop,X_trans, cudaStream);
//            cudaStream->CUDA_DEVICE_SYNCHRONIZE();
////            LOG_DEBUG("2");
//            NtsVar e_msg=torch::cat({edge_src,edge_dst},1);

            NtsVar e_msg = device_ctx[device_id]->runGraphOp<nts::op::BatchGPUSrcDstScatterOp>(sg, graph, hop, X_trans,
                                                                         cudaStream, device_id);

//            std::printf("e_msg sum: %.4lf\n", e_msg.abs().sum().item<double>());
            // Note：下面这个函数也不涉及通信
            NtsVar m = device_ctx[device_id]->runEdgeForward([&](NtsVar e_msg_){
                                             int layer = i;
                                             return torch::leaky_relu(multi_P[device_id][2 * layer + 1]->forward(e_msg_),0.2);
                                         },
                                         e_msg);//edge NN
//            std::printf("m sum: %.4lf\n", m.abs().sum().item<double>());
            //  partitioned_graph->SyncAndLog("e_msg_in");
            // Note: 下面这个函数也不涉及通信
            // a的大小为e_size，总和为src顶点的数量
            NtsVar a = device_ctx[device_id]->runGraphOp<nts::op::BatchGPUEdgeSoftMax>(sg, graph, hop,m,
                                                                                       cudaStream, device_id);// edge NN
//            std::cout << "a的行求和：\n" << a.sum(1) <<std::endl;
//            print_tensor_size(a);
//            std::printf("src size: %d, dst size: %d\n", sg->sampled_sgs[i]->src_size, sg->sampled_sgs[i]->v_size);
            cudaStream->CUDA_DEVICE_SYNCHRONIZE();
//            std::printf("a sum: %.4lf\n", a.abs().sum().item<double>());
            // Note: 下面这个函数不涉及通信
            NtsVar e_msg_out = device_ctx[device_id]->runEdgeForward([&](NtsVar a_){
//                                                     return edge_src*a_;
                                                    return e_msg.slice(1, 0, e_msg.size(1)/2, 1)*a;
                                                 },
                                                 a);//Edge NN
            //            partitioned_graph->SyncAndLog("e_msg_out");
//            std::printf("e_msg_out sum: %.4lf\n", e_msg_out.abs().sum().item<double>());
            // Note: 该函数也不涉及通信
            NtsVar nbr = device_ctx[device_id]->runGraphOp<nts::op::BatchGPUAggregateDst>(sg, graph, hop,
                                                                              e_msg_out, cudaStream, device_id);
//            std::printf("nbr sum: %.4lf\n", nbr.abs().sum().item<double>());
            // Note: 下面的函数也不涉及通信
            device_X[device_id][i+1] = device_ctx[device_id]->runVertexForward([&](NtsVar nbr_){
                return torch::relu(nbr_);
            },nbr);
//            partitioned_graph->SyncAndLog("hello 2");
        }

    }
    void Loss(NtsVar &left,NtsVar &right, nts::ctx::NtsContext* ctx) {
        torch::Tensor a = left.log_softmax(1);
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

        std::vector<VertexId> train_nids, val_nids, test_nids;
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

        std::vector<std::vector<VertexId>> multi_train_nids(num_devices);
        int start = 0;
        uint32_t per_device = train_nids.size() / num_devices;
        for(int i = 0; i < num_devices; i++) {
            int num = per_device;
            if(i == num_devices - 1) {
                num = train_nids.size() - start;
            }
            multi_train_nids[i].resize(num);
            std::memcpy(multi_train_nids[i].data(), train_nids.data() + start, num * sizeof(VertexId));
            start += num;
        }
        for(int i = 0; i < num_devices; i++) {
            nts::op::shuffle_vec(multi_train_nids[i]);
        }

//        nts::op::shuffle_vec(train_nids);
//        nts::op::shuffle_vec(val_nids);
//        nts::op::shuffle_vec(test_nids);

        InitStream();
        int layer = graph->gnnctx->layer_size.size() - 1;
        for(int i = 0; i < num_devices; i++) {
            cudaSetUsingDevice(i);
            std::printf("train id size: %d\n", multi_train_nids[i].size());
            device_train_sampler[i] = new FastSampler(fully_rep_graph,multi_train_nids[i],layer,
                                                      graph->gnnctx->fanout, pipeline_num, device_cuda_stream[i]);
            device_train_sampler[i]->set_merge_src_dst(pipeline_num);
            std::printf("sampler train id size: %d\n", device_train_sampler[i]->work_range[1]);
        }

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
                    Train(device_train_sampler[dev_id], dev_id);
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
        std::printf("run time: %.4lf (s)\n", run_time);
        std::printf("#sample time: %.4lf (s)\n", sample_time);
        std::printf("#transfer time: %.4lf (s)\n", transfer_time);
        std::printf("#train time: %.4lf (s)\n", train_time);

        delete active;
    }

    inline void InitStream() {
        for (int deviceId = 0; deviceId < num_devices; deviceId++) {
            cudaSetUsingDevice(deviceId);
            device_cuda_stream[deviceId] = new Cuda_Stream[pipeline_num];
            for (int i = 0; i < pipeline_num; i++) {
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

};

#endif //GNNMINI_GAT_SAMPLE_ALL_MULTI_HPP
