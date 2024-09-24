//
// Created by toao on 23-3-30.
//

#ifndef GNNMINI_GS_SAMPLE_CACHE_HPP
#define GNNMINI_GS_SAMPLE_CACHE_HPP
#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include <chrono>
#include <execution>
#include <tbb/concurrent_queue.h>
#include <queue>


class GS_SAMPLE_CACHE_impl {
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
    NtsVar X_mask;
    Cuda_Stream* cuda_stream;
    int pipeline_num;
    int gpu_round = 0;
    std::vector<at::cuda::CUDAStream> torch_stream;


    std::mutex sample_mutex;
    std::mutex transfer_mutex;
    std::mutex train_mutex;

    std::atomic_bool Grad_back_flag; // CPU bottom layer 等待反向梯度
    std::mutex Grad_back_mutex;
    std::condition_variable Grad_back_cv;

    std::atomic_bool W_CPU_flag; // CPU 是否求得 W 梯度 GPU端调用
    std::mutex W_CPU_mutex;
    std::condition_variable W_CPU_cv;

    std::atomic_bool W_GPU_flag; // GPU 是否求得 W 梯度 CPU端调用
    std::mutex W_GPU_mutex;
    std::condition_variable W_GPU_cv;

    uint32_t min_version = 0;   // 参数的最低版本号
    std::mutex version_mutex;
    std::condition_variable version_cv;

    // super-batch类型的PD-cache
//    tbb::concurrent_queue<ValueType*> shared_aggr_queue;
//    tbb::concurrent_queue<ValueType*> shared_embedding_queue;
//    tbb::concurrent_queue<ValueType*>  reuse_aggr_queue;
//    tbb::concurrent_queue<ValueType*> reuse_embedding_queue;
    tbb::concurrent_queue<NtsVar> shared_W_queue;


    std::atomic_flag send_back_flag = ATOMIC_FLAG_INIT;

//    std::vector<VertexId> cache_node_idx_seq;//cache 顶点选择
    // std::atomic_bool Sample_done_flag; // CPU sample 是否完成
    // std::mutex Sample_done_mutex;
    // std::condition_variable Sample_done_cv;

    SampledSubgraph *GPU_sg;
    std::thread cpu_thread;
    float cache_rate = 0.05;
    std::vector<VertexId> cache_ids;
    std::atomic_bool is_pipeline_end ;  // 流水线是否结束
    VertexId batch_cache_num;
    VertexId epoch_super_batch_num;
    VertexId last_super_batch_num;

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

    // 性能时间检测
    double gpu_wait_time = 0.0;
    double transfer_share_time = 0.0;
    double update_cache_time = 0.0;
    double cal_grad_time = 0.0;
    double cal_grad_wait_time = 0.0;
    double update_weight_time = 0.0;
    double cpu_cal_grad_time = 0.0;
    double cpu_reset_flag_time = 0.0;

    GS_SAMPLE_CACHE_impl(Graph<Empty> *graph_, int iterations_,
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
        cache_rate = graph->config->cache_rate;
        wait_times = new double[pipeline_num];
        std::printf("pipeline num: %d, cache rate: %.3f \n", pipeline_num, cache_rate);
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
//        gnndatum->init_cache_var(cache_rate);
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
            P[i]->init_pd_parameter();
            P[i]->to(GPU);
            P[i]->Adam_to_GPU();
        }

        drpmodel = torch::nn::Dropout(
                torch::nn::DropoutOptions().p(drop_rate).inplace(false));

        //        F=graph->Nts->NewOnesTensor({graph->gnnctx->l_v_num,
        //        graph->gnnctx->layer_size[0]},torch::DeviceType::CPU);

        F = graph->Nts->NewLeafTensor(
                gnndatum->local_feature,
                {static_cast<long>(graph->gnnctx->l_v_num), graph->gnnctx->layer_size[0]},
                torch::DeviceType::CPU);

        for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
            NtsVar d;
            X.push_back(d);
        }
        // X[0]=F.cuda().set_requires_grad(true);
        X[0] = F.set_requires_grad(true);

        is_pipeline_end.store(false);
        Grad_back_flag.store(false);
        W_CPU_flag.store(false);
        W_GPU_flag.store(false);
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


    void Update_GPU() {
        for (int i = 0; i < P.size(); i++) {
            // P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
//            std::printf("W[%d] grad sum: %lf\n", i, P[i]->W.grad().abs().sum().item<double>());
            P[i]->learn_local_with_decay_Adam();
            P[i]->next();
        }
    }

    NtsVar MultiplyWeight(NtsVar& a){
        int layer = graph->rtminfo->curr_layer;
        return P[layer]->forward(a);
    }
    NtsVar RunReluAndDropout(NtsVar& a) {
        return torch::dropout(torch::relu(a), drop_rate, ctx->is_train());
    }

    NtsVar vertexForward(NtsVar &a, NtsVar &x) {
        NtsVar y;
        int layer = graph->rtminfo->curr_layer;
        int layer_num = gnndatum->gnnctx->layer_size.size() - 1;
        if (layer == layer_num - 1) {
            y = P[layer]->forward(a);
            y = y.log_softmax(1); //CUDA

        } else {
            //y = P[layer]->forward(torch::relu(drpmodel(a)));
            y = torch::dropout(torch::relu(P[layer]->forward(a)), drop_rate, ctx->is_train());
        }
        return y;
    }

    /**
     * @description: 执行前向计算
     * @param {FastSampler*} sampler 训练采样器
     * @param {int} type 0：train 1：eval 2：test
     * @return {*}
     */
    void Train(FastSampler* sampler, int type=0) {
        graph->rtminfo->forward = true;
        correct = 0;
        batch = 0;
        // NtsVar target_lab;
        // X[0]=graph->Nts->NewLeafTensor({1000,F.size(1)},
        //   torch::DeviceType::CUDA);
        // target_lab=graph->Nts->NewLabelTensor({graph->config->batch_size},
        //       torch::DeviceType::CUDA);
        SampledSubgraph *sg[pipeline_num];
        NtsVar tmp_X0[pipeline_num];
        NtsVar tmp_target_lab[pipeline_num];
        for(int i = 0; i < pipeline_num; i++) {
            tmp_X0[i] = graph->Nts->NewLeafTensor({1000,F.size(1)},
                                                  torch::DeviceType::CUDA);
            tmp_target_lab[i] = graph->Nts->NewLabelTensor({graph->config->batch_size},
                                                           torch::DeviceType::CUDA);
        }
        std::thread threads[pipeline_num];
        std::vector<unsigned int> super_batch_countdown(epoch_super_batch_num); // 用于统计super batch内有多少batch已经完成
        // TODO: 最后一个super batch内batch数可能不等于pipeline数量，所以上面可以改为倒计数看看
        std::vector<bool>super_batch_ready(epoch_super_batch_num);

        for(int tid = 0; tid < pipeline_num; tid++) {
            threads[tid] = std::thread([&](int thread_id){
                wait_times[thread_id] -= get_time();
                std::unique_lock<std::mutex> sample_lock(sample_mutex, std::defer_lock);
                sample_lock.lock();
                wait_times[thread_id] += get_time();
                bool start_send_flag = false;
                int local_batch;
                int super_batch_id;
                CacheVars* cacheVars;
                NNVars* nnVars;
                while(sampler->sample_not_finished()){
                    // TODO: 在这里要确定自己的super_batch 编号，然后根据super batch编号进行确定cache
                    // 还要确定它是这个super batch里面的第几个batch，如果是第一个batch，则负责初始化工作，如果是最后一个batch，则负责收尾工作
                    local_batch = batch;
                    super_batch_id = local_batch / pipeline_num;
                    batch++;
                    // 如果是super batch的第一个batch
                    if(local_batch % pipeline_num == 0) {
                        // 初始化cache
                        cacheVars = gnndatum->new_cache_var(super_batch_id);
                        gnndatum->set_cache_index(cacheVars->cache_map, cacheVars->cache_location, super_batch_id, batch_cache_num, cache_ids);
                        super_batch_countdown[super_batch_id] = 0;
                        super_batch_ready[super_batch_id] = false;
                    } else {
                        cacheVars = gnndatum->get_cache_var(super_batch_id);
                    }
                    sample_time -= get_time();
                    //sample -1 0
                    // TODO: 下面这个要改成使用新的Cache_Map,新的Cahce_Map的判断条件也变了，变成判断super_batch是否匹配，因此函数要多添加一个super batch 参数
                    sg[thread_id]=sampler->sample_gpu_fast_omit(graph->config->batch_size, thread_id, cacheVars->dev_cache_map, super_batch_id, WeightType::Mean);
//                    sg[thread_id]=sampler->sample_gpu_fast(graph->config->batch_size, thread_id);
//                    sg=sampler->sample_fast(graph->config->batch_size);
                    gpu_round++;
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    sample_time += get_time();
                    sample_lock.unlock();


                    wait_times[thread_id] -= get_time();
                    std::unique_lock<std::mutex> transfer_lock(transfer_mutex, std::defer_lock);
                    transfer_lock.lock();
                    wait_times[thread_id] += get_time();
                    transfer_feature_time -= get_time();
                    // sampler->load_label_gpu(target_lab,gnndatum->dev_local_label);
                    // sampler->load_feature_gpu(X[0],gnndatum->dev_local_feature);
                    sampler->load_label_gpu(&cuda_stream[thread_id], sg[thread_id], tmp_target_lab[thread_id],gnndatum->dev_local_label);
                    // load feature of cacheflag -1 0, 不需要cacheflag，在sample step已经包含cacheflag信息。
                    sampler->load_feature_gpu(&cuda_stream[thread_id], sg[thread_id], tmp_X0[thread_id],gnndatum->dev_local_feature);


                    // load embedding of cacheflag = 1, cacheflag 1 to 2 CPU cache embedding to GPU cache embedding
//                    sampler->update_share_embedding(&cuda_stream[thread_id], sg[thread_id], gnndatum->dev_local_embedding,
//                                                    gnndatum->dev_share_embedding,gnndatum->dev_CacheMap,gnndatum->dev_CacheFlag);
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
//                    transfer_share_time -= get_time();
//                    sampler->update_share_embedding_and_feature(&cuda_stream[thread_id], sg[thread_id], gnndatum->dev_local_aggregation,
//                                                                gnndatum->dev_local_embedding, gnndatum->dev_share_aggregate,
//                                                                gnndatum->dev_share_embedding, gnndatum->dev_CacheMap, gnndatum->dev_CacheFlag,
//                                                                gnndatum->dev_X_version, gnndatum->dev_Y_version, require_version);
//                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
//                    transfer_share_time += get_time();
                    transfer_feature_time += get_time();
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

                    wait_times[thread_id] -= get_time();
                    std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
                    train_lock.lock();
                    wait_times[thread_id] += get_time();
                    training_time -= get_time();

                    // TODO：在下面进行提取
                    // 如果是进行训练的第一个batch，就进行提取相关数据
                    gpu_wait_time -= get_time();
                    if(super_batch_countdown[super_batch_id] == pipeline_num  || (super_batch_id == epoch_super_batch_num - 1&&
                                                                                  super_batch_countdown[super_batch_id] == last_super_batch_num )) {
                        nnVars = gnndatum->new_nn_var(super_batch_id);
//                        gnndatum->move_nn_data_to_gpu(nnVars, cuda_stream[thread_id].stream);
                    } else {
                        nnVars = gnndatum->get_nn_var(super_batch_id);
                    }
                    gpu_wait_time += get_time();



                    at::cuda::setCurrentCUDAStream(torch_stream[thread_id]);
                    Forward(sampler, tmp_X0[thread_id], sg[thread_id], &cuda_stream[thread_id], super_batch_id, cacheVars, nnVars);
                    Loss(X[graph->gnnctx->layer_size.size()-1],tmp_target_lab[thread_id]);
                    BackwardAndUpdate(sg[thread_id], &cuda_stream[thread_id], start_send_flag);
//                    P[0]->set_middle_weight();
                    training_time += get_time();
                    correct += getCorrect(X[graph->gnnctx->layer_size.size()-1], tmp_target_lab[thread_id]);
//                    batch++;
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    if((super_batch_id != epoch_super_batch_num - 1 && super_batch_countdown[super_batch_id] == pipeline_num)
                       || (super_batch_id == epoch_super_batch_num - 1 && super_batch_countdown[super_batch_id] == last_super_batch_num)) {
                        shared_W_queue.push(P[0]->W.clone());
                    }
//                    shared_W_queue.push(P[0]->W.cpu());
                    super_batch_countdown[super_batch_id]--;
                    if(super_batch_countdown[super_batch_id] == 0) {
                        // 最后一个batch了，回收相应的空间
                        gnndatum->recycle_memory(cacheVars, nnVars);
                    }

                    train_lock.unlock();
//                    P[0]->send_W_to_cpu();

                    wait_times[thread_id] -= get_time();
                    sample_lock.lock();
                    wait_times[thread_id] += get_time();

                }
                sample_lock.unlock();
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
        // loss=X[graph->gnnctx->layer_size.size()-1];
        // #run_time=1784.571165(s)
        // exec_time=1842.431756(s) reddit

        // run_time=2947.184987(s) cpu
        // exec_time=2986.859283(s)
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

    void run() {
        if (graph->partition_id == 0)
            printf("GNNmini::Engine[Dist.GPU.GCNimpl] running [%d] Epochs\n",
                   iterations);

        //      graph->print_info();
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
        // shuffle_vec(train_nids);
        // shuffle_vec(val_nids);
        // shuffle_vec(test_nids);

         nts::op::nts_local_shuffle(train_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);
        nts::op::nts_local_shuffle(val_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);
        nts::op::nts_local_shuffle(test_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);




//        sort_graph_vertex(graph->out_degree, cache_ids.data(), graph->vertices, cache_num);
        std::vector<int> fanout(1);
        fanout[0] = graph->gnnctx->fanout[graph->gnnctx->fanout.size() - 1];
        // 进行预采样
        VertexId sample_neighs = 1;
        for(int i = 0; i < graph->gnnctx->fanout.size() - 1; i++) {
            sample_neighs *= graph->gnnctx->fanout[i];
        }
        std::printf("sample last layer neighs: %u\n", sample_neighs);
        batch_cache_num = graph->config->batch_size * pipeline_num * graph->config->cache_rate * sample_neighs;
        std::printf("batch cache num: %u\n", batch_cache_num);
        auto super_batchsize = graph->config->batch_size * pipeline_num;
        auto cpu_batch_size = batch_cache_num;
        auto pre_sample_time = -get_time();
        cache_ids = preSample(train_nids, graph->config->batch_size, batch_cache_num, gnndatum->gnnctx->layer_size.size() - 1);
        pre_sample_time += get_time();
        std::printf("预采样完成，预采样时间: %lf (s)\n", pre_sample_time);
//        FastSampler* cpu_sampler = new FastSampler(graph, fully_rep_graph,
//                                                   cache_ids, 1, fanout,
//                                                   super_batchsize);
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
        auto total_super_batch_num = epoch_super_batch_num * iterations;
        // 初始化cache map这些变量
        gnndatum->init_super_batch_var(epoch_super_batch_num);
        // 首先将一个W放进相应队列里面
        shared_W_queue.push(P[0]->W.clone());
        // 使用一个多线程queue来进行存储CPU的计算结果，然后GPU会从这里面提取计算结果

        // 使用一个特定的流来进行传输
//        cudaStream_t cpu_cuda_stream;
//        cudaStreamCreateWithFlags(&cpu_cuda_stream, cudaStreamNonBlocking);
        Cuda_Stream* cudaStream_p = new Cuda_Stream();
        cudaStream_t cpu_cuda_stream = cudaStream_p->stream;



        FastSampler* cache_gpu_sampler = new FastSampler(fully_rep_graph,cache_ids,1,graph->config->batch_size,graph->gnnctx->fanout, 1, cudaStream_p);
        GPU_sg = cache_gpu_sampler->sample_gpu_fast(cpu_batch_size, WeightType::Mean);
        VertexId batch_start = 0;
        VertexId batch_end = std::min((VertexId)cache_ids.size(), batch_start + cpu_batch_size);

        // 将获得的采样子图传输到GPU
        // 建立一个变量存储feature和label
//        auto cache_gpu_label = graph->Nts->NewLeafTensor({1000,F.size(1)},
//                                                         torch::DeviceType::CUDA);
        auto cache_gpu_feature = graph->Nts->NewLeafTensor({cpu_batch_size, F.size(1)},
                                                            torch::DeviceType::CUDA);
//        cache_gpu_sampler->load_label_gpu(cudaStream_p, GPU_sg, cache_gpu_label,gnndatum->dev_local_label);
        // load feature of cacheflag -1 0, 不需要cacheflag，在sample step已经包含cacheflag信息。
        cache_gpu_sampler->load_feature_gpu(cudaStream_p, GPU_sg, cache_gpu_feature,gnndatum->dev_local_feature);

        // TODO: 编写一个无反向的GPU图操作
        auto tmpX0 = ctx->runGraphOpNoBackward<nts::op::GPUPushDownBatchOp>(GPU_sg, graph, 0,  cache_gpu_feature, batch_start, batch_end, cudaStream_p);
        NtsVar W;
        shared_W_queue.try_pop(W);
        auto y = tmpX0.matmul(W);

        // TODO: 将数据拷贝到Cache内存上

        gnndatum->move_gpu_data_to_cache_memory(tmpX0.size(0), tmpX0.size(1), y.size(1), tmpX0.packed_accessor<ValueType, 2>().data(),
                                          y.packed_accessor<ValueType, 2>().data(), cpu_cuda_stream);
        //        NtsVar mask = torch::zeros({static_cast<long>(cache_ids.size()), 1}, torch::kBool);
//        auto* mask_ptr = mask.accessor<char , 2>().data();

        double cpu_sample_time = 0.0;
        double cpu_graph_time = 0.0;
        double cpu_wait_time = 0.0;
        double cpu_nn_time = 0.0;
        double cpu_copy_time = 0.0;
        double cpu_total_time = -get_time();
        cpu_thread = std::thread([&](){
            auto max_threads = std::thread::hardware_concurrency();
            for(VertexId i = 1; i < total_super_batch_num; i++){
                if(i % epoch_super_batch_num == 0) {
                    cache_gpu_sampler->restart();
                }
                cpu_sample_time -= get_time();
                GPU_sg = cache_gpu_sampler->sample_gpu_fast(cpu_batch_size, WeightType::Mean);
                cpu_sample_time += get_time();
                // CPU采样完成，开始进行图计算
                // TODO:解决下面的batch_start和batch_end分别是什么问题
//                assert(batch_end - batch_start == GPU_sg->sampled_sgs[0]->dst().size());
                cpu_graph_time -= get_time();
                cudaStreamSynchronize(cpu_cuda_stream);
                // TODO: 将feature传输到GPU
                cache_gpu_sampler->load_feature_gpu(cudaStream_p, GPU_sg, cache_gpu_feature,gnndatum->dev_local_feature);
                tmpX0 = ctx->runGraphOpNoBackward<nts::op::GPUPushDownBatchOp>(GPU_sg, graph, 0,  cache_gpu_feature, batch_start, batch_end, cudaStream_p);
                cpu_graph_time += get_time();
                // TODO: 需要检查tmpX0是否正常，即batch，即上面的参数是否正确
                // TODO: 下面传参数的方式可能要改一下
                cpu_wait_time -= get_time();
                while(!shared_W_queue.try_pop(W)){
                    std::this_thread::yield();
                }
                cpu_wait_time += get_time();
                cpu_nn_time -= get_time();
//                while(shared_W_queue.try_pop(W)){}
//                std::printf("torch threads: %d\n", torch::get_num_threads());
//                torch::set_num_threads(31);
                torch::set_num_threads(max_threads - pipeline_num);
                y = tmpX0.mm(W);
                batch_start = batch_end % cache_ids.size();
                batch_end = std::min(batch_start + cpu_batch_size, (VertexId)cache_ids.size());
                cpu_nn_time += get_time();

                // TODO: 解决下面可能由于最后一个batch带来的memory空间不等问题
                cpu_copy_time -= get_time();
                gnndatum->move_gpu_data_to_cache_memory(tmpX0.size(0), tmpX0.size(1), y.size(1), tmpX0.packed_accessor<ValueType, 2>().data(),
                                                        y.packed_accessor<ValueType, 2>().data(), cpu_cuda_stream);
                cpu_copy_time += get_time();

            }
            cpu_total_time += get_time();
            std::printf("CPU线程已经结束\n");
            std::printf("最大线程数: %d\n", max_threads);
        });



        InitStream();

        int layer = graph->gnnctx->layer_size.size()-1;

        FastSampler* train_sampler = new FastSampler(fully_rep_graph,train_nids,layer,graph->config->batch_size,graph->gnnctx->fanout, pipeline_num, cuda_stream);
        FastSampler* eval_sampler = new FastSampler(fully_rep_graph,val_nids,layer,graph->config->batch_size,graph->gnnctx->fanout);
        FastSampler* test_sampler = new FastSampler(fully_rep_graph,test_nids,layer,graph->config->batch_size,graph->gnnctx->fanout);


        auto start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
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
            Train(train_sampler, 0);
            // ctx->eval();
            // Forward(eval_sampler, 1);
            // Forward(test_sampler, 2);
            per_epoch_time += get_time();

            std::cout << "GNNmini::Running.Epoch[" << i_i << "]:Times["
                      << per_epoch_time << "(s)]:loss\t" << loss << std::endl;
        }
        auto end_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
//        std::printf("最终的GPU version: %d\n", P[0]->gpu_version);

        exec_time += get_time();
        auto wait_time = 0.0;
        for(int i = 0; i < pipeline_num; i++) {
            wait_time += wait_times[i];
        }
        printf("#run_time=%lf(s)\n", exec_time);
        printf("all:%lf(s) prepro:%lf(s) pro:%lf(s) post:%lf(s) copy:%lf(s)\n",train_sampler->all_time,train_sampler->pre_pro_time, train_sampler->pro_time,train_sampler->post_pro_time,train_sampler->copy_gpu_time );
        printf("test_time:%lf(s)\n",train_sampler->test_time);
        printf("#wait time: %lf(s)\n", wait_time);
        printf("#gather_feature_time=%lf(s)\n", gather_feature_time);
        std::printf("cpu inclusiveTime: %lf\n", train_sampler->cs->cpu_inclusiveTime);
        std::printf("inclusiveTime: %lf\n", train_sampler->cs->inclusiveTime);
        std::printf("init layer time: %lf\n", train_sampler->init_layer_time);
        std::printf("init co time: %lf\n", train_sampler->init_co_time);
        std::printf("transfer share time: %lf (s)\n", transfer_share_time);
        std::printf("update cache time: %lf (s)\n", update_cache_time);

        // CPU相关时间
        std::printf("cpu_total_time: %.4lf (s)\n", cpu_total_time);
        std::printf("cpu_sample_time: %.4lf (s)\n", cpu_sample_time);
        std::printf("cpu_graph_time: %.4lf (s)\n", cpu_graph_time);
        std::printf("cpu_wait_time: %.4lf (s)\n", cpu_wait_time);
        std::printf("cpu_nn_time: %.4lf (s)\n", cpu_nn_time);
        std::printf("cpu_copy_time: %.4lf (s)\n", cpu_copy_time);

        std::printf("# gpu wait time: %.4lf (s)\n", gpu_wait_time);
        std::printf("# gpu round: %d\n", gpu_round);
        std::printf("# pre sample time: %lf (s)\n", pre_sample_time);
        printf("#sample_time= %.4lf (s)\n", (sample_time));
        printf("#transfer_feature_time= %.4lf (s)\n", (transfer_feature_time));
        printf("#training_time= %.4lf (s)\n", training_time);
        delete active;
        cpu_thread.join();
        printf("#average epoch time: %lf\n", exec_time/iterations);
//        printf("总采样数:%llu, 总命中数:%llu\n", Cuda_Stream::total_sample_num, Cuda_Stream::total_cache_hit);
//        printf("平均epoch采样数:%llu, 平均epoch命中数:%llu\n", Cuda_Stream::total_sample_num/iterations, Cuda_Stream::total_cache_hit/iterations);
        printf("总传输节点数: %llu\n", Cuda_Stream::total_transfer_node);
//        printf("平均epoch传输节点数:%llu\n", Cuda_Stream::total_transfer_node/iterations);
        printf("%lu\n%lu\n", start_time, end_time);

    }

    // TODO: 目前presample只考虑了一阶邻居，适用于两层网络的情况，需要进行下一步改进
    std::vector<VertexId> preSample(std::vector<VertexId>& train_ids, int batch_size, int batch_cache_num, int layers) {
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
                              &(batch_cache_ids[batch_cache_num * i]), layers);
        }
        return batch_cache_ids;
    }

    void get_most_neighbor(std::vector<VertexId>& train_ids, VertexId start, VertexId super_batch_size, VertexId batch_cache_num,
                           VertexId* cache_arr_start, int layers) {
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

//        for(VertexId id = start; id < start + super_batch_size && id < train_ids.size(); id++) {
//            auto col_start = fully_rep_graph->column_offset[train_ids[id]];
//            auto col_end = fully_rep_graph->column_offset[train_ids[id]+1];
//            for(auto i = col_start; i < col_end; i++){
//                auto neighbor = fully_rep_graph->row_indices[i];
//                // 进行宽度优先遍历
//                std::queue<VertexId> neighbor_queue;
//                neighbor_queue.push(neighbor);
//                for(int layer = 2; layer < layers; layer++) {
//                    size_t queue_num = neighbor_queue.size();
//                    for(size_t neis = 0; neis < queue_num; neis++){
//                        auto node = neighbor_queue.front();
//                        neighbor_queue.pop();
//                        auto node_col_start = fully_rep_graph->column_offset[node];
//                        auto node_col_end = fully_rep_graph->column_offset[node+1];
//                        for(auto new_node = node_col_start; new_node < node_col_end; new_node++){
//                            neighbor_queue.push(fully_rep_graph->row_indices[new_node]);
//                        }
//                    }
//                }
//                // 统计neighbor queue里面的内容
//                while(!neighbor_queue.empty()) {
//                    auto node = neighbor_queue.front();
//                    neighbor_queue.pop();
//
//                    // 统计相应节点的数量
//                    if(sample_count_map.find(node) == sample_count_map.end()){
//                        sample_count_map[node] = 0;
//                    } else {
//                        sample_count_map[node]++;
//                    }
//                }
//
//            }
//        }
//
//        // 下面是对map进行排序
//        std::vector<std::pair<VertexId, VertexId>> pairs;
//        for (auto itr = sample_count_map.begin(); itr != sample_count_map.end(); ++itr){
//            pairs.push_back(*itr);
//        }
//        std::sort(pairs.begin(), pairs.end(), [=](std::pair<VertexId, VertexId>& a, std::pair<VertexId, VertexId>& b){
//                      return a.second > b.second;
//                  }
//        );
//        // TODO: 这里如果不够的话随便用一些节点来填
//        for(VertexId i = 0; i < batch_cache_num && i < pairs.size(); i++) {
//            cache_arr_start[i] = pairs[i].first;
//        }

    }

    //code by aix from 406 to 434
    void CacheFlag_init(float Cacherate){
        std::vector<VertexId> cache_node_idx_seq;
        cache_node_idx_seq.resize(graph->vertices);
        std::iota(cache_node_idx_seq.begin(), cache_node_idx_seq.end(), 0);
        std::sort(std::execution::par ,cache_node_idx_seq.begin(), cache_node_idx_seq.end(), [&](const int x, const int y) {
            return graph->out_degree_for_backward[x] > graph->out_degree_for_backward[y];
        });

        int max_threads = std::thread::hardware_concurrency();

#pragma omp parallel for num_threads(max_threads)
        for (int i = 0; i < graph->vertices; ++i) {
            gnndatum->CacheFlag[i] = -1; //init
            gnndatum->CacheMap[i] = -1;
        }

        int cache_node_num = graph->vertices * Cacherate;
#pragma omp parallel for num_threads(max_threads)
        for (int i = 0; i < cache_node_num; ++i) {
            // LOG_DEBUG("cache_nodes[%d] = %d", i, cache_nodes[i]);
            gnndatum->CacheFlag[cache_node_idx_seq[i]] = 0; //初始化为cache顶点
            gnndatum->CacheMap[cache_node_idx_seq[i]] = i;
        }

        cache_ids.resize(cache_node_num);
        std::copy(cache_node_idx_seq.begin(), cache_node_idx_seq.begin() + cache_node_num, cache_ids.begin());
    }

    inline void Forward(FastSampler* sampler, NtsVar& tmp_X0, SampledSubgraph* sg, Cuda_Stream* cudaStream,
                        VertexId super_batch_id, CacheVars* cacheVars, NNVars* nnVars){
        for(int l = 0; l < (graph->gnnctx->layer_size.size()-1); l++){//forward
            graph->rtminfo->curr_layer = l;
            int hop = (graph->gnnctx->layer_size.size()-2) - l;
            if(l == 0) {
                // Note: 下面用于debug
//                sampler->print_avg_weight(cudaStream, sg, gnndatum->dev_CacheFlag);

                NtsVar Y_i=ctx->runGraphOp<nts::op::SingleGPUAllSampleGraphOp>(sg,graph,hop,tmp_X0,cudaStream);
//                sampler->load_share_aggregate(cudaStream, sg,gnndatum->dev_share_aggregate,
//                                                          Y_i, gnndatum->CacheMap,gnndatum->CacheFlag);

                X[l + 1] = ctx->runVertexForward(
                        [&](NtsVar n_i,NtsVar v_i){
                            auto Y_W = MultiplyWeight(n_i);
                            cudaStream->CUDA_DEVICE_SYNCHRONIZE();
                            update_cache_time -= get_time();
                            sampler->load_share_embedding_and_feature(cudaStream, sg,nnVars->dev_shared_aggr, nnVars->dev_shared_embedding,
                                                                      Y_i, Y_W, cacheVars->dev_cache_map,cacheVars->dev_cache_location, super_batch_id);
                            cudaStream->CUDA_DEVICE_SYNCHRONIZE();
                            update_cache_time += get_time();
//                                         Y_sum = Y_i.abs().sum().item<double>();
//                                        std::printf("after Y row: %d, sum: %lf, avg: %lf\n", Y_i.size(0), Y_sum, Y_sum/Y_i.size(0));
                            return RunReluAndDropout(Y_W);
                        },
                        Y_i,
                        tmp_X0
                );
            } else {
                NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUAllSampleGraphOp>(sg,graph,hop,X[l],cudaStream);
                X[l + 1] = ctx->runVertexForward([&](NtsVar n_i,NtsVar v_i){
                                                     return vertexForward(n_i, v_i);
                                                 },
                                                 Y_i,
                                                 X[l]);
            }
        }
    }

    inline void BackwardAndUpdate(SampledSubgraph* sg, Cuda_Stream* cudaStream, bool& start_send_flag){
        if (ctx->training) {
            ctx->self_backward(false);
            Update_GPU();

            for(int i = 0; i < P.size(); i++) {
                P[i]->zero_grad();
            }
        }
    }


//    void CacheFlga_refresh(int Cacherate){
//        int cache_node_num = graph->vertices * Cacherate;
//        for (int i = 0; i < cache_node_num; ++i) {
//            // LOG_DEBUG("cache_nodes[%d] = %d", i, cache_nodes[i]);
//            gnndatum->CacheFlag[cache_node_idx_seq[i]] = 0; //初始化为cache顶点
//        }
//    }
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

#endif //GNNMINI_GS_SAMPLE_CACHE_HPP
