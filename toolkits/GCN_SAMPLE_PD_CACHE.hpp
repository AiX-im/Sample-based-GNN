//
// Created by toao on 23-3-30.
//

#ifndef GNNMINI_GCN_SAMPLE_PD_CACHE_HPP
#define GNNMINI_GCN_SAMPLE_PD_CACHE_HPP
#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include <chrono>
#include <execution>
#include <tbb/concurrent_queue.h>
#include <queue>


class GCN_SAMPLE_PD_CACHE_impl {
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
    VertexId super_batchsize;
    VertexId top_cache_num;
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

    // for feature cache
    std::vector<int> cache_node_idx_seq;
    VertexId* cache_node_hashmap;
    VertexId* dev_cache_node_hashmap;
    int cache_node_num = 0;
    float* dev_cache_feature;
    VertexId *local_idx, *local_idx_cache, *dev_local_idx, *dev_local_idx_cache;
    // VertexId *dev_cache_cnt, *local_cache_cnt;
    VertexId *outmost_vertex;
    double used_gpu_mem, total_gpu_mem;


    FastSampler* train_sampler = nullptr;
    SampledSubgraph *CPU_sg;
    std::thread cpu_thread;
    float cache_rate = 0.05;
    std::vector<VertexId> cache_ids;
    std::atomic_bool is_pipeline_end ;  // 流水线是否结束
    std::vector<VertexId> batch_cache_num;
    VertexId epoch_super_batch_num;
    VertexId last_super_batch_num;
    std::vector<VertexId> train_nids, val_nids, test_nids;

    NtsVar Y_PD;
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

    GCN_SAMPLE_PD_CACHE_impl(Graph<Empty> *graph_, int iterations_,
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
//                        gnndatum->set_cache_index(cacheVars->cache_map, cacheVars->cache_location, super_batch_id, batch_cache_num, cache_ids);
                        gnndatum->set_cache_index(cacheVars->cache_map, cacheVars->cache_location, super_batch_id, batch_cache_num, cache_ids);
                        super_batch_countdown[super_batch_id] = 0;
                        super_batch_ready[super_batch_id] = false;
                    } else {
                        cacheVars = gnndatum->get_cache_var(super_batch_id);
                    }
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    sample_time -= get_time();
                    //sample -1 0
                    // TODO: 下面这个要改成使用新的Cache_Map,新的Cahce_Map的判断条件也变了，变成判断super_batch是否匹配，因此函数要多添加一个super batch 参数
                    sg[thread_id]=sampler->sample_gpu_fast_omit(graph->config->batch_size, thread_id, cacheVars->dev_cache_map, super_batch_id);
//                    sg[thread_id]=sampler->sample_gpu_fast(graph->config->batch_size, thread_id);
//                    sg=sampler->sample_fast(graph->config->batch_size);
                    gpu_round++;
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    sample_time += get_time();
//                    std::printf("sample time: %.4lf\n", sample_time);
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
                    // sampler->load_feature_gpu(&cuda_stream[thread_id], sg[thread_id], tmp_X0[thread_id],gnndatum->dev_local_feature);

                    if (!graph->config->cacheflag) {  // trans feature use zero copy (omit gather feature)
                        sampler->load_feature_gpu(&cuda_stream[thread_id], sg[thread_id],
                         tmp_X0[thread_id],gnndatum->dev_local_feature);                        
                        // get_gpu_mem(used_gpu_mem, total_gpu_mem);
                    } else {  // trans freature which is not cache in gpu
                        // epoch_transfer_feat_time -= get_time();
                            sampler->load_feature_gpu_cache(
                            &cuda_stream[thread_id], sg[thread_id], tmp_X0[thread_id], gnndatum->dev_local_feature,
                            dev_cache_feature, local_idx, local_idx_cache, cache_node_hashmap, dev_local_idx,
                            dev_local_idx_cache, dev_cache_node_hashmap, outmost_vertex);
                    }


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
                    correct += getCorrect(X[graph->gnnctx->layer_size.size()-1], tmp_target_lab[thread_id]);
//                    batch++;
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    training_time += get_time();
                    if((super_batch_id != epoch_super_batch_num - 1 && super_batch_countdown[super_batch_id] == pipeline_num)
                       || (super_batch_id == epoch_super_batch_num - 1 && super_batch_countdown[super_batch_id] == last_super_batch_num)) {
                        shared_W_queue.push(P[0]->W.cpu());
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

        std::printf("train:%d\n",train_nids.size());
        std::printf("val_nids:%d\n",val_nids.size());
        std::printf("test_nids:%d\n",test_nids.size());

        InitStream();

        int layer = graph->gnnctx->layer_size.size()-1;

        train_sampler = new FastSampler(fully_rep_graph,train_nids,layer,graph->gnnctx->fanout, pipeline_num, cuda_stream);
        // FastSampler* eval_sampler = new FastSampler(fully_rep_graph,val_nids,layer,graph->gnnctx->fanout);
        // FastSampler* test_sampler = new FastSampler(fully_rep_graph,test_nids,layer,graph->gnnctx->fanout);

        initCacheVariable();
        determine_cache_node_idx(graph->vertices * graph->config->feature_cache_rate);
        LOG_DEBUG("feature_cache_rate %f", graph->config->feature_cache_rate);
        //layer 0 graphop push down code by aix from 566 to 573

        long batch_sum = 0;
        for(int i = 0; i < batch_cache_num.size(); i++) {
            batch_sum += batch_cache_num[i];
        }
        std::printf("before shuffle sum: %ld, total cache: %ld\n", batch_sum, cache_ids.size());

//         nts::op::nts_local_shuffle(train_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);
//        nts::op::nts_local_shuffle(val_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);
//        nts::op::nts_local_shuffle(test_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);
        nts::op::nts_local_shuffle(train_nids,  graph->config->batch_size * pipeline_num, cache_ids, batch_cache_num);

        batch_sum = 0;
        for(int i = 0; i < batch_cache_num.size(); i++) {
            batch_sum += batch_cache_num[i];
        }
        std::printf("before shuffle sum: %ld, total cache: %ld\n", batch_sum, cache_ids.size());

        

//        sort_graph_vertex(graph->out_degree, cache_ids.data(), graph->vertices, cache_num);
//        // 进行预采样
//        VertexId sample_neighs = 1;
//        for(int i = 0; i < graph->gnnctx->fanout.size() - 1; i++) {
//            sample_neighs *= graph->gnnctx->fanout[i];
//        }
//        std::printf("sample last layer neighs: %u\n", sample_neighs);
//        batch_cache_num = graph->config->batch_size * pipeline_num * cache_rate * sample_neighs;
//        std::printf("batch cache num: %u\n", batch_cache_num);
//        auto super_batchsize = graph->config->batch_size * pipeline_num;
//        auto cpu_batch_size = batch_cache_num;
//        auto pre_sample_time = -get_time();
//        cache_ids = preSample(train_nids, graph->config->batch_size, batch_cache_num, gnndatum->gnnctx->layer_size.size() - 1);
//        pre_sample_time += get_time();
//        std::printf("预采样完成，预采样时间: %lf (s)\n", pre_sample_time);
        std::vector<int> fanout(1);
        fanout[0] = graph->gnnctx->fanout[graph->gnnctx->fanout.size() - 1];
        FastSampler* cpu_sampler = new FastSampler(graph, fully_rep_graph,
                                                   cache_ids, 1, fanout,
                                                   super_batchsize);
//        // 计算CPU循环次数
//        // CPU计算的次数可由epoch的数量和一个epoch内super batch的数量确定
//        epoch_super_batch_num = train_nids.size() / super_batchsize;
//        last_super_batch_num = train_nids.size() - super_batchsize * epoch_super_batch_num;
//        last_super_batch_num = std::ceil((double)last_super_batch_num/(double)graph->config->batch_size);
//        if(last_super_batch_num != 0) {
//            epoch_super_batch_num++;
//        } else {
//            last_super_batch_num = pipeline_num;
//        }
        auto total_super_batch_num = epoch_super_batch_num * iterations;
        // 初始化cache map这些变量
        gnndatum->init_super_batch_var(epoch_super_batch_num);
        // 首先将一个W放进相应队列里面
        shared_W_queue.push(P[0]->W.cpu());
        // 使用一个多线程queue来进行存储CPU的计算结果，然后GPU会从这里面提取计算结果

        // 使用一个特定的流来进行传输
        cudaStream_t cpu_cuda_stream;
        cudaStreamCreateWithFlags(&cpu_cuda_stream, cudaStreamNonBlocking);


        CPU_sg = cpu_sampler->sample_fast(batch_cache_num[0]);
        // CPU_sg = cpu_sampler->sample_fast_allPD(batch_cache_num[0]);
        VertexId batch_start = 0;
        VertexId batch_end = std::min((VertexId)cache_ids.size(), batch_start + batch_cache_num[0]);
        // CPU会按行进行聚合，所以行需要有序   %Y_PD 全部顶点的一阶全邻居聚合结果
        auto tmpX0 = ctx->runGraphOpNoBackward<nts::op::PushDownBatchOp>(CPU_sg, graph, 0,  F, batch_start, batch_end);
        // std::printf("load_aggresult start \n");
        // auto tmpX0 = cpu_sampler->load_aggresult(Y_PD);
        // std::printf("load_aggresult end \n");
        
        NtsVar W;
        shared_W_queue.try_pop(W);
        auto y = tmpX0.matmul(W);
//        gnndatum->move_data_to_pin_memory(tmpX0.size(0), tmpX0.size(1), y.size(1), tmpX0.accessor<ValueType, 2>().data(),
//                                          y.accessor<ValueType, 2>().data(), cpu_cuda_stream);
        gnndatum->move_embedding_to_gpu(top_cache_num, y.size(0), y.size(1), y.accessor<ValueType, 2>().data(),
                                        cpu_cuda_stream);
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
                // TODO: 如果报错，就是这里转移的问题
                batch_start = batch_end % cache_ids.size();
                batch_end = std::min(batch_start + batch_cache_num[i%epoch_super_batch_num], (VertexId)cache_ids.size());

                if(i % epoch_super_batch_num == 0) {
                    cpu_sampler->restart();
                }
                cpu_sample_time -= get_time();
                CPU_sg = cpu_sampler->sample_fast(batch_cache_num[i%epoch_super_batch_num]);
                // CPU_sg = cpu_sampler->sample_fast_allPD(batch_cache_num[i%epoch_super_batch_num]);

                cpu_sample_time += get_time();
                // CPU采样完成，开始进行图计算
                // TODO:解决下面的batch_start和batch_end分别是什么问题
                assert(batch_cache_num[i%epoch_super_batch_num] == CPU_sg->sampled_sgs[0]->dst().size());
                // assert(batch_end - batch_start == CPU_sg->sampled_sgs[0]->dst().size());
                cpu_graph_time -= get_time();
                tmpX0 = ctx->runGraphOpNoBackward<nts::op::PushDownBatchOp>(CPU_sg, graph, 0, F, 0, batch_cache_num[i%epoch_super_batch_num]);
                // auto tmpX0 = cpu_sampler->load_aggresult(Y_PD);
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
                cpu_nn_time += get_time();

                // TODO: 解决下面可能由于最后一个batch带来的memory空间不等问题
                cpu_copy_time -= get_time();
                gnndatum->move_embedding_to_gpu(top_cache_num, y.size(0), y.size(1), y.accessor<ValueType, 2>().data(),
                                                cpu_cuda_stream);
                cpu_copy_time += get_time();

            }
            cpu_total_time += get_time();
            std::printf("CPU线程已经结束\n");
            std::printf("最大线程数: %d\n", max_threads);
        });


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


        std::printf("cpu inclusiveTime: %.4lf (s)\n", train_sampler->cs->cpu_inclusiveTime);
        std::printf("inclusiveTime: %.4lf (s)\n", train_sampler->cs->inclusiveTime);
        std::printf("init layer time: %.4lf (s)\n", train_sampler->init_layer_time);
        std::printf("init co time: %.4lf (s)\n", train_sampler->init_co_time);
        std::printf("pro time: %.4lf (s)\n", train_sampler->pro_time);
        std::printf("post pro time: %.4lf (s)\n", train_sampler->post_pro_time);

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
        printf("#sample_time= %.4lf (s)\n", (sample_time));
        printf("#transfer_feature_time= %.4lf (s)\n", (transfer_feature_time));
        printf("#training_time= %.4lf (s)\n", training_time);
        delete active;
        cpu_thread.join();
        printf("#average epoch time: %lf\n", exec_time/iterations);
       printf("总采样数:%llu, 总命中数:%llu\n", Cuda_Stream::total_sample_num, Cuda_Stream::total_cache_hit);
       printf("平均epoch采样数:%llu, 平均epoch命中数:%llu\n", Cuda_Stream::total_sample_num/iterations, Cuda_Stream::total_cache_hit/iterations);
        printf("总传输节点数: %llu\n", Cuda_Stream::total_transfer_node);
//        printf("平均epoch传输节点数:%llu\n", Cuda_Stream::total_transfer_node/iterations);
        printf("%lu\n%lu\n", start_time, end_time);

    }


    inline void Forward(FastSampler* sampler, NtsVar& tmp_X0, SampledSubgraph* sg, Cuda_Stream* cudaStream,
                        VertexId super_batch_id, CacheVars* cacheVars, NNVars* nnVars){
//         print_tensor_size(tmp_X0);
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
//                            sampler->load_share_embedding_and_feature(cudaStream, sg,nnVars->dev_shared_aggr, nnVars->dev_shared_embedding,
//                                                                      Y_i, Y_W, cacheVars->dev_cache_map,cacheVars->dev_cache_location, super_batch_id);

                            sampler->load_share_embedding(cudaStream, sg, nnVars->dev_shared_embedding,
                                                          Y_W, cacheVars->dev_cache_map,
                                                          cacheVars->dev_cache_location, super_batch_id);
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
        std::printf("edge_filename: %s\n", graph->config->edge_file.c_str());
//        cache_ids = nts::op::preSample(train_nids, graph->config->batch_size, batch_cache_num,
//                                       gnndatum->gnnctx->layer_size.size() - 1, fully_rep_graph,
//                                       pipeline_num);

        cache_ids = nts::op::preSample(train_nids, graph->config->batch_size, batch_cache_num, cache_rate, top_cache_num,
                                       gnndatum->gnnctx->layer_size.size() - 1, fully_rep_graph,
                                       cache_rate / 0.8, graph,  pipeline_num);
        pre_sample_time += get_time();
        std::printf("预采样时间: %.4lf, cache num: %ld\n", pre_sample_time, cache_ids.size());


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

    }

 void mark_cache_node(std::vector<int>& cache_nodes) {
    // init mask
    // #pragma omp parallel for
    // #pragma omp parallel for num_threads(threads)
    // for (int i = 0; i < graph->vertices; ++i) {
    //   cache_node_hashmap[i] = -1;
    //   // assert(cache_node_hashmap[i] == -1);
    // }

    // mark cache nodes
    int tmp_idx = 0;
    for (int i = 0; i < cache_node_num; ++i) {
      // LOG_DEBUG("cache_nodes[%d] = %d", i, cache_nodes[i]);
      cache_node_hashmap[cache_nodes[i]] = tmp_idx++;
    }
    LOG_DEBUG("cache_node_num %d tmp_idx %d", cache_node_num, tmp_idx);
    assert(cache_node_num == tmp_idx);
  }

void cache_high_degree(std::vector<int>& node_idx) {
    std::sort(node_idx.begin(), node_idx.end(), [&](const int x, const int y) {
      return graph->out_degree_for_backward[x] > graph->out_degree_for_backward[y];
    });
    // #pragma omp parallel for num_threads(threads)
    // for (int i = 1; i < graph->vertices; ++i) {
    //   assert(graph->out_degree_for_backward[node_idx[i]] <= graph->out_degree_for_backward[node_idx[i - 1]]);
    // }
    mark_cache_node(node_idx);
  }

void determine_cache_node_idx(int node_nums) {
    if (node_nums > graph->vertices) 
        node_nums = graph->vertices;
    cache_node_num = node_nums;
    LOG_DEBUG("cache_node_num %d (%.3f)", cache_node_num, 1.0 * cache_node_num / graph->vertices);

    // LOG_DEBUG("start get_gpu_idle_mem()");
    // double max_gpu_mem = get_gpu_idle_mem();
    LOG_DEBUG("start get_gpu_idle_mem_pipe()");
    double max_gpu_mem = get_gpu_idle_mem_pipe();
    LOG_DEBUG("release gpu memory");
    empty_gpu_cache();
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    LOG_DEBUG("used %.3f total %.3f (after emptyCache)", used_gpu_mem, total_gpu_mem);
    // double free_memory = total_gpu_mem - max_gpu_mem - 200;
    double free_memory = total_gpu_mem - max_gpu_mem - 100;
    int memory_nodes = free_memory * 1024 * 1024 / sizeof(ValueType) / graph->gnnctx->layer_size[0];
    cache_node_num = memory_nodes;
    LOG_DEBUG("cache_node_num %d (%.3f)", cache_node_num, 1.0 * cache_node_num / graph->vertices);
    if (cache_node_num > graph->vertices) 
        cache_node_num = graph->vertices;

    cache_node_idx_seq.resize(graph->vertices);
    std::iota(cache_node_idx_seq.begin(), cache_node_idx_seq.end(), 0);
    // cache_node_hashmap.resize(graph->vertices);
    cache_node_hashmap = (VertexId*)cudaMallocPinned(1ll * graph->vertices * sizeof(VertexId));
    dev_cache_node_hashmap = (VertexId*)getDevicePointer(cache_node_hashmap);

    // #pragma omp parallel for
    // #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < graph->vertices; ++i) {
      cache_node_hashmap[i] = -1;
      // assert(cache_node_hashmap[i] == -1);
    }

    cache_high_degree(cache_node_idx_seq);

    gater_cpu_cache_feature_and_trans_to_gpu();
  }

void gater_cpu_cache_feature_and_trans_to_gpu() {
    long feat_dim = graph->gnnctx->layer_size[0];
    dev_cache_feature = (float*)cudaMallocGPU(cache_node_num * sizeof(float) * feat_dim);
    // gather_cache_feature, prepare trans to gpu
    // LOG_DEBUG("start gather_cpu_cache_feature");
    float* local_cache_feature_gather = new float[cache_node_num * feat_dim];
// std::cout << "###" << cache_node_num * sizeof(float) * feat_dim << " " << cache_node_num * feat_dim << std::endl;
// std::cout << "###" << graph->vertices * feat_dim << "l_v_num " << graph->gnnctx->l_v_num << " " << graph->vertices <<
// std::endl;
// #pragma omp parallel for
// omp_set_num_threads(threads);
auto max_threads = std::thread::hardware_concurrency();
#pragma omp parallel for num_threads(max_threads)
    for (int i = 0; i < cache_node_num; ++i) {
      int node_id = cache_node_idx_seq[i];
      // assert(node_id < graph->vertices);
      // assert(node_id < graph->gnnctx->l_v_num);
      // LOG_DEBUG("copy node_id %d to", node_id);
      // LOG_DEBUG("local_id %d", cache_node_hashmap[node_id]);

      for (int j = 0; j < feat_dim; ++j) {
        assert(cache_node_hashmap[node_id] < cache_node_num);
        local_cache_feature_gather[cache_node_hashmap[node_id] * feat_dim + j] =
            gnndatum->local_feature[node_id * feat_dim + j];
      }
    }
    LOG_DEBUG("start trans to gpu");
    move_data_in(dev_cache_feature, local_cache_feature_gather, 0, cache_node_num, feat_dim);
    local_idx = (VertexId*)cudaMallocPinned(1ll * graph->vertices * sizeof(VertexId));
    local_idx_cache = (VertexId*)cudaMallocPinned(1ll * graph->vertices * sizeof(VertexId));
    outmost_vertex = (VertexId*)malloc(graph->vertices * sizeof(VertexId));
    // local_cache_cnt = (VertexId*)cudaMallocPinned(3 * sizeof(VertexId));
    // dev_cache_cnt = (VertexId*)getDevicePointer(local_cache_cnt);
    dev_local_idx = (VertexId*)getDevicePointer(local_idx);
    dev_local_idx_cache = (VertexId*)getDevicePointer(local_idx_cache);
  }

     // pre train some epochs to get idle memory of GPU when training
  double get_gpu_idle_mem_pipe() {
        double max_gpu_used = 0;
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
        for(int tid = 0; tid < pipeline_num; tid++) {
            threads[tid] = std::thread([&](int thread_id){
                std::unique_lock<std::mutex> sample_lock(sample_mutex, std::defer_lock);
                sample_lock.lock();
                while(train_sampler->sample_not_finished()){
                    sg[thread_id]=train_sampler->sample_gpu_fast(graph->config->batch_size, thread_id);
                    //sg=sampler->sample_fast(graph->config->batch_size);
                    cudaStreamSynchronize(cuda_stream[thread_id].stream);
                    sample_lock.unlock();

                    std::unique_lock<std::mutex> transfer_lock(transfer_mutex, std::defer_lock);
                    transfer_lock.lock();
                    train_sampler->load_label_gpu(&cuda_stream[thread_id], sg[thread_id], tmp_target_lab[thread_id],gnndatum->dev_local_label);
                    train_sampler->load_feature_gpu(&cuda_stream[thread_id], sg[thread_id], tmp_X0[thread_id],gnndatum->dev_local_feature);
                    cudaStreamSynchronize(cuda_stream[thread_id].stream);
                    transfer_lock.unlock();
                    std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
                    train_lock.lock();
                    at::cuda::setCurrentCUDAStream(torch_stream[thread_id]);
                    for(int l = 0; l < (graph->gnnctx->layer_size.size()-1); l++){//forward
                        graph->rtminfo->curr_layer = l;
                        int hop = (graph->gnnctx->layer_size.size()-2) - l;
                        if(l == 0) {
                            NtsVar Y_i=ctx->runGraphOp<nts::op::SingleGPUAllSampleGraphOp>(sg[thread_id],graph,hop,tmp_X0[thread_id],&cuda_stream[thread_id]);
                            X[l + 1] = ctx->runVertexForward([&](NtsVar n_i,NtsVar v_i){
                                                                 return vertexForward(n_i, v_i);
                                                             },
                                                             Y_i,
                                                             tmp_X0[thread_id]);
                        } else {
                            NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUAllSampleGraphOp>(sg[thread_id],graph,hop,X[l],&cuda_stream[thread_id]);
                            X[l + 1] = ctx->runVertexForward([&](NtsVar n_i,NtsVar v_i){
                                                                 return vertexForward(n_i, v_i);
                                                             },
                                                             Y_i,
                                                             X[l]);
                        }
                    }

                    Loss(X[graph->gnnctx->layer_size.size()-1],tmp_target_lab[thread_id]);
                    if (ctx->training) {
                        ctx->self_backward(false);
                        for (int i = 0; i < P.size(); i++) {
                            P[i]->zero_grad();
                        }
                    }
                    cudaStreamSynchronize(cuda_stream[thread_id].stream);
                    train_lock.unlock();
                    sample_lock.lock();

                }
                sample_lock.unlock();
            }, tid);
        }
        for(int i = 0; i < pipeline_num; i++) {
            threads[i].join();
        }
        train_sampler->restart();

        get_gpu_mem(used_gpu_mem, total_gpu_mem);
        max_gpu_used = std::max(used_gpu_mem, max_gpu_used);
        LOG_DEBUG("get_gpu_idle_mem_pipe(): used %.3f max_used %.3f total %.3f", used_gpu_mem, max_gpu_used,
                    total_gpu_mem);

        return max_gpu_used;
    }

    void get_gpu_mem(double &used, double &total) {
            int deviceCount = 0;
            cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
            if (deviceCount == 0) {
                std::cout << "当前PC没有支持CUDA的显卡硬件设备" << std::endl;
                assert(false);
            }

            size_t gpu_total_size;
            size_t gpu_free_size;

            cudaError_t cuda_status = cudaMemGetInfo(&gpu_free_size, &gpu_total_size);

            if (cudaSuccess != cuda_status) {
                std::cout << "Error: cudaMemGetInfo fails : " << cudaGetErrorString(cuda_status) << std::endl;
                assert(false);
                // gpu_free_size = 0, gpu_total_size = 0;
            }

            double total_memory = double(gpu_total_size) / (1024.0 * 1024.0);
            double free_memory = double(gpu_free_size) / (1024.0 * 1024.0);
            double used_memory = total_memory - free_memory;
            used = used_memory;
            total = total_memory;
            // return {used_memory, total_memory};
            // std::cout << "\n"
            //     << "当前显卡总共有显存" << total_memory << "m \n"
            //     << "已使用显存" << used_memory << "m \n"
            //     << "剩余显存" << free_memory << "m \n" << std::endl;
    }
    void empty_gpu_cache() {
        for (int ti = 0; ti < 5; ++ti) {  // clear gpu cache memory
        c10::cuda::CUDACachingAllocator::emptyCache();
        }
    }

};

#endif //GNNMINI_GCN_SAMPLE_PD_CACHE_HPP
