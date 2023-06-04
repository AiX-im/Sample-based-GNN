//
// Created by toao on 23-3-30.
//

#ifndef GNNMINI_GS_CACHE_TEST_HPP
#define GNNMINI_GS_CACHE_TEST_HPP
#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include <chrono>
#include <execution>
class GS_CACHE_TEST_impl {
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


    std::atomic_flag send_back_flag = ATOMIC_FLAG_INIT;

//    std::vector<VertexId> cache_node_idx_seq;//cache 顶点选择
    // std::atomic_bool Sample_done_flag; // CPU sample 是否完成
    // std::mutex Sample_done_mutex;
    // std::condition_variable Sample_done_cv;

    SampledSubgraph *CPU_sg;
    std::thread cpu_thread;
    float cache_rate = 0.05;
    std::vector<VertexId> cache_ids;
    std::atomic_bool is_pipeline_end ;  // 流水线是否结束

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
    double transfer_share_time = 0.0;
    double update_cache_time = 0.0;
    double cal_grad_time = 0.0;
    double cal_grad_wait_time = 0.0;
    double update_weight_time = 0.0;
    double cpu_cal_grad_time = 0.0;
    double cpu_reset_flag_time = 0.0;

    GS_CACHE_TEST_impl(Graph<Empty> *graph_, int iterations_,
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
        gnndatum->init_cache_var(cache_rate);
        if (0 == graph->config->feature_file.compare("random")) {
            gnndatum->random_generate();
        } else {
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
        if (layer == 1) {
            y = P[layer]->forward(a);
            y = y.log_softmax(1); //CUDA

        } else if (layer == 0) {
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
        for(int tid = 0; tid < pipeline_num; tid++) {
            threads[tid] = std::thread([&](int thread_id){
                wait_times[thread_id] -= get_time();
                std::unique_lock<std::mutex> sample_lock(sample_mutex, std::defer_lock);
                sample_lock.lock();
                wait_times[thread_id] += get_time();
                bool start_send_flag = false;
                while(sampler->sample_not_finished()){
                    sample_time -= get_time();
                    //sample -1 0
                    sg[thread_id]=sampler->sample_gpu_fast_omit(graph->config->batch_size, thread_id, gnndatum->dev_CacheFlag, WeightType::Mean);
//                    sg[thread_id]=sampler->sample_gpu_fast(graph->config->batch_size, thread_id);
                    //sg=sampler->sample_fast(graph->config->batch_size);
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
                    auto gpu_version = P[0]->gpu_version;
                    uint32_t require_version = gpu_version < pipeline_num ? 0 : gpu_version - pipeline_num;
//                    std::printf("min version: %d, gpu version: %d, require version: %d\n", min_version, gpu_version, require_version);
//                    if(gpu_version > pipeline_num + min_version) {
////                        std::printf("CPU的版本: %d, 需要的版本: %d，进入等待\n", min_version, gpu_version);
//                        std::unique_lock<std::mutex> version_lock(version_mutex);
//                        version_cv.wait(version_lock, [&]{return gpu_version  <= pipeline_num + min_version;});
//                    }
                    // Note: 这里是用于测试看一下使用最新版本的会怎样
//                    if(gpu_version != min_version) {
////                        std::printf("CPU的版本太旧，进入等待\n");
//                        std::unique_lock<std::mutex> version_lock(version_mutex);
//                        version_cv.wait(version_lock, [&]{return P[0]->gpu_version == min_version;});
//                    }
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    transfer_share_time -= get_time();
                    gnndatum->set_gpu_transfer_flag();
                    sampler->update_share_embedding_and_feature(&cuda_stream[thread_id], sg[thread_id], gnndatum->dev_local_aggregation,
                                                                gnndatum->dev_local_embedding, gnndatum->dev_share_aggregate,
                                                                gnndatum->dev_share_embedding, gnndatum->dev_CacheMap, gnndatum->dev_CacheFlag,
                                                                gnndatum->dev_X_version, gnndatum->dev_Y_version, require_version);
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    gnndatum->unset_gpu_transfer_flag();
                    transfer_share_time += get_time();
                    transfer_feature_time += get_time();
              transfer_lock.unlock();

                    wait_times[thread_id] -= get_time();
                    std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
                    train_lock.lock();
                    wait_times[thread_id] += get_time();
                    training_time -= get_time();

                    at::cuda::setCurrentCUDAStream(torch_stream[thread_id]);
                    Forward(sampler, tmp_X0[thread_id], sg[thread_id], &cuda_stream[thread_id]);
                    Loss(X[graph->gnnctx->layer_size.size()-1],tmp_target_lab[thread_id]);
                    BackwardAndUpdate(sg[thread_id], &cuda_stream[thread_id], start_send_flag);
                    P[0]->set_middle_weight();
                    training_time += get_time();
                    correct += getCorrect(X[graph->gnnctx->layer_size.size()-1], tmp_target_lab[thread_id]);
                    batch++;
                    cuda_stream[thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    train_lock.unlock();
                    P[0]->send_W_to_cpu();

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
        shuffle_vec(train_nids);
        shuffle_vec(val_nids);
        shuffle_vec(test_nids);


        CacheFlag_init(cache_rate);
        // 下面是开始进行CPU的任务
        // get cache vertexs
        VertexId cache_num = graph->vertices * cache_rate;
        gnndatum->cache_num = cache_num;
        P[0]->init_shared_grad_buffer(cache_num, gnndatum->gnnctx->layer_size[1]);
        // Note: 这里可以修改cpu采样的batch
        VertexId cpu_batch_size = 64;
        std::vector<ValueType> X0_ptr(cache_num * gnndatum->gnnctx->layer_size[1]);
//        NtsVar X0;
        std::printf("cache num: %d\n", cache_num);

        VertexId range_num = (cache_num/cpu_batch_size) + 1;
        std::vector<VertexId> range_arr(range_num);
        std::iota(range_arr.begin(), range_arr.end(), 0u);

        auto* cache_ids_ptr = cache_ids.data();
//        sort_graph_vertex(graph->out_degree, cache_ids.data(), graph->vertices, cache_num);
        std::vector<int> fanout(1);
        fanout[0] = graph->gnnctx->fanout[graph->gnnctx->fanout.size() - 1];
        FastSampler* cpu_sampler = new FastSampler(graph, fully_rep_graph,
                                                   cache_ids, 1, fanout,
                                                   cache_num);
        CPU_sg = cpu_sampler->sample_fast(cache_num, WeightType::Mean);
        VertexId batch_start = 0;
        VertexId batch_end = std::min(cache_num, batch_start + cpu_batch_size);
        // CPU会按行进行聚合，所以行需要有序
        auto tmpX0 = ctx->runGraphOpNoBackward<nts::op::PushDownBatchOp>(CPU_sg, graph, 0,  F, batch_start, batch_end);
//        NtsVar mask = torch::zeros({static_cast<long>(cache_ids.size()), 1}, torch::kBool);
//        auto* mask_ptr = mask.accessor<char , 2>().data();
        cpu_thread = std::thread([&](){
            uint32_t W_version;
            uint32_t current_min_version = std::numeric_limits<uint32_t>::max();
            uint32_t epoch_min_version = 0;
            auto cpu_sample_time = 0.0;
            auto cpu_cal_time = 0.0;
            while(!is_pipeline_end.load()){
//                // 进行nn计算
//                auto W = P[0]->get_W_and_version(W_version);
////                std::printf("X size:(%d, %d), W size: (%d, %d), version: %d\n", tmpX0.size(0), tmpX0.size(1), W.size(0), W.size(1), W_version);
//                auto y = tmpX0.matmul(W);
//                current_min_version = gnndatum->X_version[batch_start];
//                if(current_min_version > min_version){
//                    min_version = current_min_version;
//                    version_cv.notify_all();
//                }
////               std::printf("tmp X0 sum: %lf\n", tmpX0.sum().item<double>());
//
//                // 将结果传到gpu
////                gnndatum->move_data_to_local_cache(y.size(0), y.size(1), y.accessor<ValueType , 2>().data(),
////                                                   &(cache_ids_ptr[batch_start]), mask_ptr, batch_start);
////                auto y_sum = y.sum().abs().item<float>();
////                std::printf("CPU y row: %d, y sum: %f, y average: %f\n", y.size(0), y_sum, y_sum/(y.size(0)));
////                auto F_sum = F.sum().abs().item<float>();
////                std::printf("F row: %d, F sum: %f, F avg: %f\n", F.size(0), F_sum, F_sum/(F.size(0) * F.size(1)));
////                auto X_sum = tmpX0.abs().sum().item<float>();
//                gnndatum->move_data_to_local_cache(y.size(0), gnndatum->gnnctx->layer_size[0],
//                                                   y.size(1), tmpX0.accessor<ValueType, 2>().data(),
//                                                   y.accessor<ValueType, 2>().data(), &(cache_ids_ptr[batch_start]),
//                                                   batch_start, W_version);
//
//
//                // 检查是否到最后，计算完了的话重启采样器，并进行采样, 然后进行反向
//                if(batch_end == cache_num) {
//                    cpu_sampler->restart();
//                    CPU_sg = cpu_sampler->sample_fast(cache_num);
////                   std::printf("before mask X0 dim: %d\n", X0.dim());
////                   std::printf("before mask X0 size: (%d, %d)\n", X0.size(0), X0.size(1));
////                   std::printf("mask size: (%d, %d)\n", mask.size(0), mask.size(1));
//
//                }
//                batch_start = batch_end % cache_num;
//                batch_end = std::min(cache_num, batch_start + cpu_batch_size);
//
//                // 进行CPU的聚合
//                tmpX0 = ctx->runGraphOpNoBackward<nts::op::PushDownBatchOp>(CPU_sg, graph, 0,  F, batch_start, batch_end);

//                auto W = P[0]->get_W_and_version(W_version);
//                current_min_version = W_version;
                cpu_cal_time -= get_time();
                std::for_each(std::execution::seq, range_arr.begin(), range_arr.end(), [&](VertexId& arr_offset){
                    VertexId batch_start = arr_offset *  cpu_batch_size;
                    VertexId batch_end = std::min(cache_num, batch_start + cpu_batch_size);
                    if(batch_start == batch_end) {
                        return;
                    }
                    auto tmpX0 = ctx->runGraphOpNoBackward<nts::op::PushDownBatchOp>(CPU_sg, graph, 0,  F, batch_start, batch_end);
                    // 进行nn计算
                    auto W = P[0]->get_W_and_version(W_version);
                    auto y = tmpX0.matmul(W);
//                    if(current_min_version > min_version){
//                        min_version = current_min_version;
//                        version_cv.notify_all();
//                    }
                    gnndatum->move_data_to_local_cache(y.size(0), gnndatum->gnnctx->layer_size[0],
                                                       y.size(1), tmpX0.accessor<ValueType, 2>().data(),
                                                       y.accessor<ValueType, 2>().data(), &(cache_ids_ptr[batch_start]),
                                                       batch_start, W_version);

                    auto batch_min_version = gnndatum->X_version[batch_start];
                    write_min(&current_min_version, batch_min_version);

                });
                cpu_cal_time += get_time();
                cpu_sample_time -= get_time();
//                std::cout << "current_min_version: " << current_min_version << ", gpu version: " << P[0]->gpu_version << ", middle version: " << P[0]->middle_version << std::endl;
                gnndatum->try_swap_buffer();
                if(current_min_version > min_version) {
                    min_version = current_min_version;
//                    version_cv.notify_all();
                }
                version_cv.notify_all();
                current_min_version = std::numeric_limits<uint32_t>::max();
                cpu_sampler->restart();
                CPU_sg = cpu_sampler->sample_fast(cache_num, WeightType::Mean);
                cpu_sample_time += get_time();
            }
            std::printf("#cpu sample time: %lf\n", cpu_sample_time);
            std::printf("#cpu cal time: %lf\n", cpu_cal_time);
        });



        InitStream();

        int layer = graph->gnnctx->layer_size.size()-1;

        FastSampler* train_sampler = new FastSampler(fully_rep_graph,train_nids,layer,graph->gnnctx->fanout, pipeline_num, cuda_stream);
        FastSampler* eval_sampler = new FastSampler(fully_rep_graph,val_nids,layer,graph->gnnctx->fanout);
        FastSampler* test_sampler = new FastSampler(fully_rep_graph,test_nids,layer,graph->gnnctx->fanout);

//        long in_degree_sum = 0;
//        long out_degree_sum = 0;
//#pragma omp parallel for reduction(+: in_degree_sum, out_degree_sum)
//        for(long i = 0; i < graph->vertices; i++){
//            in_degree_sum += graph->in_degree_for_backward[i];
//            out_degree_sum += graph->out_degree_for_backward[i];
//        }
//        std::printf("in_degree_sum: %lu, out_degree_sum: %lu\n", in_degree_sum, out_degree_sum);

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
        printf("#sample_time=%lf(s)\n", (sample_time));
        printf("#transfer_feature_time=%lf(s)\n", (transfer_feature_time));
        printf("#training_time=%lf(s)\n", training_time);
        printf("#gather_feature_time=%lf(s)\n", gather_feature_time);
        std::printf("cpu inclusiveTime: %lf\n", train_sampler->cs->cpu_inclusiveTime);
        std::printf("inclusiveTime: %lf\n", train_sampler->cs->inclusiveTime);
        std::printf("init layer time: %lf\n", train_sampler->init_layer_time);
        std::printf("init co time: %lf\n", train_sampler->init_co_time);
        std::printf("transfer share time: %lf (s)\n", transfer_share_time);
        std::printf("update cache time: %lf (s)\n", update_cache_time);
        delete active;
        is_pipeline_end.store(true);
        Grad_back_flag.store(true);
        W_CPU_flag.store(true);
        W_GPU_flag.store(true);
        W_CPU_cv.notify_all();
        W_GPU_cv.notify_all();
        Grad_back_cv.notify_all();
        cpu_thread.join();

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

    inline void Forward(FastSampler* sampler, NtsVar& tmp_X0, SampledSubgraph* sg, Cuda_Stream* cudaStream){
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
//                                        auto X_mask = sampler->get_X_mask(cudaStream, sg, gnndatum->dev_CacheFlag);
//                                        auto mask_sum = X_mask.sum().item<int>();
//                                        std::printf("X mask sum: %d\n", X_mask.sum().item<int>());
//                                        auto cache_tensor = torch::masked_select(Y_i, X_mask);
//                                        cache_tensor = cache_tensor.reshape({mask_sum, Y_i.size(1)});
//                                        auto cache_sum = cache_tensor.abs().sum().item<double>();
//                                        std::printf("before cache aggregate row: %d, sum: %lf, avg: %lf\n", cache_tensor.size(0), cache_sum, cache_sum/cache_tensor.size(0));
//
//                                        auto Y_sum = Y_i.abs().sum().item<double>();
//                                        std::printf("before Y row: %d, sum: %lf, avg: %lf\n", Y_i.size(0), Y_sum, Y_sum/Y_i.size(0));
                                        cudaStream->CUDA_DEVICE_SYNCHRONIZE();
                                        update_cache_time -= get_time();
                                         sampler->load_share_embedding_and_feature(cudaStream, sg,gnndatum->dev_share_aggregate, gnndatum->dev_share_embedding,
                                                                                   Y_i, Y_W, gnndatum->dev_CacheMap,gnndatum->dev_CacheFlag);
                                         cudaStream->CUDA_DEVICE_SYNCHRONIZE();
                                         update_cache_time += get_time();
//                                         Y_sum = Y_i.abs().sum().item<double>();
//                                        std::printf("after Y row: %d, sum: %lf, avg: %lf\n", Y_i.size(0), Y_sum, Y_sum/Y_i.size(0));
                                         return RunReluAndDropout(Y_W);
                                     },
                                     Y_i,
                                     tmp_X0
                                     );
                //load embedding of cacheflag = 2 or 3 to X[l + 1], GPU cache embedding
//                X_mask = torch::zeros({X[l+1].size(0), 1}, at::TensorOptions().dtype(torch::kBool).device_index(0));
//                sampler->load_share_embedding(cudaStream, sg, gnndatum->dev_share_embedding, X[l + 1],gnndatum->CacheMap,
//                                              gnndatum->CacheFlag, X_mask, gnndatum->dev_mask_tensor);

//                sampler->load_share_embedding(cudaStream, sg, gnndatum->dev_share_embedding, X[l + 1],gnndatum->CacheMap,gnndatum->CacheFlag);
//                std::this_thread::sleep_for(std::chrono::milliseconds(500));
//                cudaDeviceSynchronize();
//                auto X1_sum = X[1].sum().item<double>();
//                std::printf("X1数量: %d, X1 sum: %lf, X1 avg: %lf\n", X[1].size(0), X1_sum, X1_sum/X[1].size(0));
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

#endif //GNNMINI_GS_CACHE_TEST_HPP
