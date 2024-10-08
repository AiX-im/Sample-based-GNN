//
// Created by toao on 23-9-17.
//

#ifndef GNNMINI_GAT_SAMPLE_ALL_GPU_HPP
#define GNNMINI_GAT_SAMPLE_ALL_GPU_HPP
#include "core/neutronstar.hpp"
class GAT_SAMPLE_ALL_GPU_impl {
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
    NtsVar MASK_gpu;
    //GraphOperation *gt;
    PartitionedGraph *partitioned_graph;
    // Variables
    std::vector<Parameter *> P;
    std::vector<NtsVar> X;
    nts::ctx::NtsContext* ctx;

    NtsVar F;
    NtsVar loss;
    NtsVar tt;
    int batch;
    long correct;
    int pipeline_num;
    VertexId batch_cache_num;
    VertexId epoch_super_batch_num;
    VertexId last_super_batch_num;
    std::mutex sample_mutex;
    std::mutex transfer_mutex;
    std::mutex train_mutex;
    std::vector<VertexId> cache_ids;
    int gpu_round = 0;
    Cuda_Stream* cuda_stream;
    double gpu_wait_time = 0.0;
    std::vector<at::cuda::CUDAStream> torch_stream;
    tbb::concurrent_queue<NtsVar> shared_W_queue;


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

    GAT_SAMPLE_ALL_GPU_impl(Graph<Empty> *graph_, int iterations_,
                      bool process_local = false, bool process_overlap = false) {
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
        torch::manual_seed(0);
        gnndatum = new GNNDatum(graph->gnnctx, graph);
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
        gnndatum->genereate_gpu_data();
        L_GT_G = L_GT_C.cuda();
        MASK_gpu = MASK.cuda();

        // initializeing parameter. Creating tensor with shape [layer_size[i],
        // layer_size[i + 1]]
        for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
            P.push_back(new Parameter(graph->gnnctx->layer_size[i],
                                      graph->gnnctx->layer_size[i + 1], alpha, beta1,
                                      beta2, epsilon, weight_decay));
            P.push_back(new Parameter(graph->gnnctx->layer_size[i + 1] * 2, 1, alpha,
                                      beta1, beta2, epsilon, weight_decay));
        }

        // synchronize parameter with other processes
        // because we need to guarantee all of workers are using the same model
        torch::Device GPU(torch::kCUDA, 0);
        for (int i = 0; i < P.size(); i++) {
            P[i]->init_parameter();
            P[i]->set_decay(decay_rate, decay_epoch);
            P[i]->to(GPU);
            P[i]->Adam_to_GPU();
        }

        F = graph->Nts->NewLeafTensor(
                gnndatum->local_feature,
                {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
                torch::DeviceType::CPU);

        NtsVar d;
        X.resize(graph->gnnctx->layer_size.size(),d);
        // X[0] is the initial vertex representation. We created it from
        // local_feature
        // X[0] = F.cuda();
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
            if (s == 0) {
                LOG_INFO("Train Acc: %f %d %d", acc_train, g_train, g_correct);
            } else if (s == 1) {
                LOG_INFO("Eval Acc: %f %d %d", acc_train, g_train, g_correct);
            } else if (s == 2) {
                LOG_INFO("Test Acc: %f %d %d", acc_train, g_train, g_correct);
            }
        }
    }
    void Loss() {
        //  return torch::nll_loss(a,L_GT_C);
        torch::Tensor a = X[graph->gnnctx->layer_size.size() - 1].log_softmax(1);
        torch::Tensor mask_train = MASK_gpu.eq(0);
        loss = torch::nll_loss(
                a.masked_select(mask_train.expand({mask_train.size(0), a.size(1)}))
                        .view({-1, a.size(1)}),
                L_GT_G.masked_select(mask_train.view({mask_train.size(0)})));
        ctx->appendNNOp(X[graph->gnnctx->layer_size.size() - 1], loss);
    }

    void Update() {
        for (int i = 0; i < P.size(); i++) {
//            std::printf("layer %d W gradient sum: %.4lf\n", i, P[i]->W.grad().abs().sum().item<double>());
            // accumulate the gradient using all_reduce
            P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
            // update parameters with Adam optimizer
            P[i]->learnC2G_with_decay_Adam();
            P[i]->next();
        }
    }

    void Train(FastSampler* sampler, int type = 0) {
        graph->rtminfo->forward = true;
        correct = 0;
        batch = 0;

        SampledSubgraph *sg[pipeline_num];
        NtsVar tmp_X0[pipeline_num];
        NtsVar tmp_target_lab[pipeline_num];

        for(int i = 0; i < pipeline_num; i++) {
            tmp_X0[i] = graph->Nts->NewLeafTensor({graph->config->batch_size,F.size(1)},
                                                  torch::DeviceType::CUDA);
            tmp_target_lab[i] = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
        }

        std::thread threads[pipeline_num];
        std::vector<unsigned int> super_batch_countdown(epoch_super_batch_num);
        std::vector<unsigned int> super_batch_ready(epoch_super_batch_num);

        for(int tid = 0; tid < pipeline_num; tid++) {
            threads[tid] = std::thread([&](int thread_id) {
                std::unique_lock<std::mutex> sample_lock(sample_mutex, std::defer_lock);
                sample_lock.lock();
                int local_batch;

                while(sampler->sample_not_finished()) {
                    sample_time -= get_time();
                    batch++;
                    // 如果是super batch的第一个batch
                    sg[thread_id] = sampler->sample_gpu_fast(graph->config->batch_size, thread_id,
                                                             WeightType::None);
                    gpu_round++;
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    sample_time += get_time();
                    sample_lock.unlock();

                    std::unique_lock<std::mutex> transfer_lock(transfer_mutex, std::defer_lock);
                    transfer_lock.lock();
                    transfer_time -= get_time();
                    sampler->load_label_gpu(&cuda_stream[thread_id], sg[thread_id], tmp_target_lab[thread_id],gnndatum->dev_local_label);
                    // load feature of cacheflag -1 0, 不需要cacheflag，在sample step已经包含cacheflag信息。
                    sampler->load_feature_gpu(&cuda_stream[thread_id], sg[thread_id], tmp_X0[thread_id],gnndatum->dev_local_feature);
//                    sampler->load_dst_src_feature_gpu(&cuda_stream[thread_id], sg[thread_id], tmp_X0[thread_id], gnndatum->dev_local_feature);
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    transfer_time += get_time();
                    transfer_lock.unlock();

                    std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
                    train_lock.lock();
                    train_time -= get_time();

                    at::cuda::setCurrentCUDAStream(torch_stream[thread_id]);
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    Forward(sampler, tmp_X0[thread_id], sg[thread_id], &cuda_stream[thread_id]);
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    Loss(X[graph->gnnctx->layer_size.size()-1],tmp_target_lab[thread_id]);
                    BackwardAndUpdate();
                    correct += getCorrect(X[graph->gnnctx->layer_size.size()-1], tmp_target_lab[thread_id]);
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    train_time += get_time();
                    train_lock.unlock();

                    sample_lock.lock();
                }
                sample_lock.unlock();

            }, tid);
        }

        // 多流等待
        for(int i = 0; i < pipeline_num; i++) {
            threads[i].join();
        }
        sampler->restart();
        auto acc = 1.0 * correct / sampler->work_range[1];
        if (type == 0) {
            LOG_INFO("Train Acc: %f %d %d", acc, correct, sampler->work_range[1]);
        } else if (type == 1) {
            LOG_INFO("Eval Acc: %f %d %d", acc, correct, sampler->work_range[1]);
        } else if (type == 2) {
            LOG_INFO("Test Acc: %f %d %d", acc, correct, sampler->work_range[1]);
        }

    }
    inline void Forward(FastSampler* sampler, NtsVar& tmp_X0, SampledSubgraph* sg, Cuda_Stream* cudaStream){

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
                X_trans=ctx->runVertexForward([&](NtsVar x_i_){
                                                  int layer = graph->rtminfo->curr_layer;
                                                  return P[2 * layer]->forward(x_i_);
                                              },
                                              tmp_X0);
            } else {
                // 计算W*h，乘的是上面的第一个，相当于GCN的vertexForward
                X_trans=ctx->runVertexForward([&](NtsVar x_i_){
                                                  int layer = graph->rtminfo->curr_layer;
                                                  return P[2 * layer]->forward(x_i_);
                                              },
                                              X[i]);
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

            NtsVar e_msg = ctx->runGraphOp<nts::op::BatchGPUSrcDstScatterOp>(sg, graph, hop, X_trans,
                                                                         cudaStream);

//            std::printf("e_msg sum: %.4lf\n", e_msg.abs().sum().item<double>());
            // Note：下面这个函数也不涉及通信
            NtsVar m=ctx->runEdgeForward([&](NtsVar e_msg_){
                                             int layer = graph->rtminfo->curr_layer;
                                             return torch::leaky_relu(P[2 * layer + 1]->forward(e_msg_),0.2);
                                         },
                                         e_msg);//edge NN
//            std::printf("m sum: %.4lf\n", m.abs().sum().item<double>());
            //  partitioned_graph->SyncAndLog("e_msg_in");
            // Note: 下面这个函数也不涉及通信
            // a的大小为e_size，总和为src顶点的数量
            NtsVar a=ctx->runGraphOp<nts::op::BatchGPUEdgeSoftMax>(sg, graph, hop,m, cudaStream);// edge NN
//            std::cout << "a的行求和：\n" << a.sum(1) <<std::endl;
//            print_tensor_size(a);
//            std::printf("src size: %d, dst size: %d\n", sg->sampled_sgs[i]->src_size, sg->sampled_sgs[i]->v_size);
            cudaStream->CUDA_DEVICE_SYNCHRONIZE();
//            std::printf("a sum: %.4lf\n", a.abs().sum().item<double>());
            // Note: 下面这个函数不涉及通信
            NtsVar e_msg_out=ctx->runEdgeForward([&](NtsVar a_){
//                                                     return edge_src*a_;
                                                    return e_msg.slice(1, 0, e_msg.size(1)/2, 1)*a;
                                                 },
                                                 a);//Edge NN
            //            partitioned_graph->SyncAndLog("e_msg_out");
//            std::printf("e_msg_out sum: %.4lf\n", e_msg_out.abs().sum().item<double>());
            // Note: 该函数也不涉及通信
            NtsVar nbr= ctx->runGraphOp<nts::op::BatchGPUAggregateDst>(sg, graph, hop,e_msg_out, cudaStream);
//            std::printf("nbr sum: %.4lf\n", nbr.abs().sum().item<double>());
            // Note: 下面的函数也不涉及通信
            X[i+1]=ctx->runVertexForward([&](NtsVar nbr_){
                return torch::relu(nbr_);
            },nbr);
//            partitioned_graph->SyncAndLog("hello 2");
        }

    }
    void Loss(NtsVar &left,NtsVar &right) {
        torch::Tensor a = left.log_softmax(1);
        loss = torch::nll_loss(a,right);
        if (ctx->training == true) {
            ctx->appendNNOp(left, loss);
        }
    }

    inline void BackwardAndUpdate(){
        if(ctx->training) {
            ctx->self_backward(false);
            Update();

            for(int i = 0; i < P.size(); i++) {
                P[i]->zero_grad();
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

        nts::op::shuffle_vec(train_nids);
        nts::op::shuffle_vec(val_nids);
        nts::op::shuffle_vec(test_nids);

        InitStream();
        int layer = graph->gnnctx->layer_size.size() - 1;
        FastSampler* train_sampler = new FastSampler(fully_rep_graph,train_nids,layer,graph->config->batch_size,graph->gnnctx->fanout, pipeline_num, cuda_stream);
        train_sampler->set_merge_src_dst(pipeline_num);

        exec_time -= get_time();
        auto run_time = -get_time();
        for (int i_i = 0; i_i < iterations; i_i++) {
            double per_epoch_time = -get_time();
            graph->rtminfo->epoch = i_i;
            if (i_i != 0) {
                for (int i = 0; i < P.size(); i++) {
                    P[i]->zero_grad();
                }
            }
            Train(train_sampler, 0);
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
            if (graph->partition_id == 0)
                std::cout << "Nts::Running.Epoch[" << i_i << "]:Times["<< per_epoch_time
                            << "(s)]:loss\t" << loss.item<float>() << std::endl;
        }
        run_time += get_time();
        std::printf("run time: %.4lf (s)\n", run_time);
        std::printf("#sample time: %.4lf (s)\n", sample_time);
        std::printf("#transfer time: %.4lf (s)\n", transfer_time);
        std::printf("#train time: %.4lf (s)\n", train_time);

        delete active;
    }

    inline void InitStream() {
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
    }

};

#endif //GNNMINI_GAT_SAMPLE_ALL_GPU_HPP
