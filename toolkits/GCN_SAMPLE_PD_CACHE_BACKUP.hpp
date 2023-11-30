//
// Created by toao on 23-3-30.
//

#ifndef GNNMINI_GCN_SAMPLE_PD_CACHE_HPP
#define GNNMINI_GCN_SAMPLE_PD_CACHE_HPP
#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
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

    std::atomic_flag send_back_flag = ATOMIC_FLAG_INIT;

//    std::vector<VertexId> cache_node_idx_seq;//cache 顶点选择
    // std::atomic_bool Sample_done_flag; // CPU sample 是否完成
    // std::mutex Sample_done_mutex;
    // std::condition_variable Sample_done_cv;

    SampledSubgraph *CPU_sg;
    std::thread cpu_thread;
    float cache_rate = 0.1;
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
    double update_time = 0.0;
    double grad_back_time = 0.0;
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
        gnndatum->init_cache_var(cache_rate);
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

    void Update(bool synchronize = false) {
        for (int i = 0; i < P.size(); i++) {
            // P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
            if(synchronize && i == 0) {
                // 将梯度和W传到CPU
                grad_back_time -= get_time();
                P[0]->send_param_to_cpu();
                // 唤醒CPU等待的线程，即CPU可能等待GPU传回梯度
                Grad_back_flag.store(true);
                Grad_back_cv.notify_all();
                grad_back_time += get_time();

                // 计算GPU的梯度并传回CPU
                cal_grad_time -= get_time();
                P[0]->cal_GPU_gradient();
                // 通知CPU GPU的梯度已经计算完成
                W_GPU_flag.store(true);
                W_GPU_cv.notify_all();

                // 如果CPU的未完成，则GPU进行等待
                if(!W_CPU_flag.load()) {
                    cal_grad_wait_time -= get_time();
                    std::unique_lock<std::mutex> lk(W_CPU_mutex);
                    W_CPU_cv.wait(lk, [&](){return W_CPU_flag.load();});
                    cal_grad_wait_time += get_time();
                }
                cal_grad_time += get_time();

                // 用完之后立即进行重置，防止一起重置的同步开销
                update_weight_time -= get_time();
                W_CPU_flag.store(false);
                P[0]->learn_gpu_with_decay_Adam();
                update_weight_time += get_time();
            } else {
                P[i]->learn_local_with_decay_Adam();
            }
            P[i]->next();
        }
    }

    void SendGradToCpu(){
        // 将梯度和W传到CPU
        grad_back_time -= get_time();
        P[0]->send_param_to_cpu();
        // 唤醒CPU等待的线程，即CPU可能等待GPU传回梯度
        Grad_back_flag.store(true);
        Grad_back_cv.notify_all();
        grad_back_time += get_time();
    }

    void UpdateCache(bool synchronize = false) {
        for (int i = P.size() - 1; i >= 0; i--) {
            // P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
            if(synchronize && i == 0) {
                // 计算GPU的梯度并传回CPU
                cal_grad_time -= get_time();
                P[0]->cal_GPU_gradient();
                // 通知CPU GPU的梯度已经计算完成
                W_GPU_flag.store(true);
                W_GPU_cv.notify_all();

                // 如果CPU的未完成，则GPU进行等待
                if(!W_CPU_flag.load()) {
                    cal_grad_wait_time -= get_time();
                    std::unique_lock<std::mutex> lk(W_CPU_mutex);
                    W_CPU_cv.wait(lk, [&](){return W_CPU_flag.load();});
                    cal_grad_wait_time += get_time();
                }
                cal_grad_time += get_time();

                // 用完之后立即进行重置，防止一起重置的同步开销
                update_weight_time -= get_time();
                W_CPU_flag.store(false);
                P[0]->learn_gpu_with_decay_Adam();
                update_weight_time += get_time();
            } else {
                P[i]->learn_local_with_decay_Adam();
            }
            P[i]->next();
        }
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
                    sg[thread_id]=sampler->sample_gpu_fast_omit(graph->config->batch_size, thread_id, gnndatum->CacheFlag);
                    //sg=sampler->sample_fast(graph->config->batch_size);
                    cudaStreamSynchronize(cuda_stream[thread_id].stream);
                    sample_time += get_time();
                    sample_lock.unlock();

//              wait_times[thread_id] -= get_time();
//              std::unique_lock<std::mutex> transfer_lock(transfer_mutex, std::defer_lock);
//              transfer_lock.lock();
//              wait_times[thread_id] += get_time();
                    transfer_feature_time -= get_time();
                    // sampler->load_label_gpu(target_lab,gnndatum->dev_local_label);
                    // sampler->load_feature_gpu(X[0],gnndatum->dev_local_feature);
                    sampler->load_label_gpu(&cuda_stream[thread_id], sg[thread_id], tmp_target_lab[thread_id],gnndatum->dev_local_label);
                    // load feature of cacheflag -1 0, 不需要cacheflag，在sample step已经包含cacheflag信息。
                    sampler->load_feature_gpu(&cuda_stream[thread_id], sg[thread_id], tmp_X0[thread_id],gnndatum->dev_local_feature);

                    // load embedding of cacheflag = 1, cacheflag 1 to 2 CPU cache embedding to GPU cache embedding
                    sampler->update_share_embedding(&cuda_stream[thread_id], sg[thread_id], gnndatum->dev_local_embedding,
                                                    gnndatum->dev_share_embedding,gnndatum->dev_CacheMap,gnndatum->dev_CacheFlag);
                    cudaStreamSynchronize(cuda_stream[thread_id].stream);
                    transfer_feature_time += get_time();
//              transfer_lock.unlock();

                    wait_times[thread_id] -= get_time();
                    std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
                    train_lock.lock();
                    wait_times[thread_id] += get_time();
                    training_time -= get_time();


                    at::cuda::setCurrentCUDAStream(torch_stream[thread_id]);
                    Forward(sampler, tmp_X0[thread_id], sg[thread_id], &cuda_stream[thread_id]);
                    Loss(X[graph->gnnctx->layer_size.size()-1],tmp_target_lab[thread_id]);
                    BackwardAndUpdate(sg[thread_id], &cuda_stream[thread_id], start_send_flag);
                    cudaStreamSynchronize(cuda_stream[thread_id].stream);
                    training_time += get_time();
                    correct += getCorrect(X[graph->gnnctx->layer_size.size()-1], tmp_target_lab[thread_id]);
                    batch++;
                    train_lock.unlock();

                    CheckFlagAndSendGrad(start_send_flag);



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

    inline void ExecCPUBackward(NtsVar& X, NtsVar& mask) {
        // 新的W和梯度是否传回
        // 没传回进行等待
        if(!Grad_back_flag.load()){
            std::unique_lock<std::mutex> lk(Grad_back_mutex);
            Grad_back_cv.wait(lk, [&]{return Grad_back_flag.load();});
            if(is_pipeline_end.load()){
                return;
            }
        }
        cpu_cal_grad_time -= get_time();
        Grad_back_flag.store(false);

        // GPU梯度传回之后，就重置cache flag
        cpu_reset_flag_time -= get_time();
        gnndatum->reset_cache_flag(cache_ids);
        cpu_reset_flag_time += get_time();

        // 传回了计算相应的梯度
        P[0]->cal_CPU_gradient(X, mask);
        cpu_cal_grad_time += get_time();

        // 检查GPU的梯度是否完成
        // 如果是CPU先计算完成了，则进行等待，如果是GPU先计算完成了，则直接进行下一步
        // 不过gpu那边也需要进行等待, 前面这里包含了交换的过程
        // 唤醒GPU端线程
        W_CPU_flag.store(true);
        W_CPU_cv.notify_all();
        if(!W_GPU_flag.load()){
            std::unique_lock<std::mutex> lk(W_GPU_mutex);
            W_GPU_cv.wait(lk, [&]{return W_GPU_flag.load();});
            if(is_pipeline_end.load()){
                return;
            }
        }
        W_GPU_flag.store(false);
        // 累加GPU的梯度
        P[0]->reduce_GPU_gradient();

        // 更新参数W
        P[0]->learn_cpu_with_decay_Adam();
        // next在GPU端调用即可
//        P[0]->next();


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
        shuffle_vec(train_nids);
        shuffle_vec(val_nids);
        shuffle_vec(test_nids);


        CacheFlag_init(cache_rate);
        // 下面是开始进行CPU的任务
        // get cache vertexs
        VertexId cache_num = graph->vertices * cache_rate;
        gnndatum->cache_num = cache_num;
        P[0]->init_shared_grad_buffer(cache_num, gnndatum->gnnctx->layer_size[1]);
        VertexId cpu_batch_size = 128;
        std::vector<ValueType> X0_ptr(cache_num * gnndatum->gnnctx->layer_size[1]);
        NtsVar X0;

        auto* cache_ids_ptr = cache_ids.data();
//        sort_graph_vertex(graph->out_degree, cache_ids.data(), graph->vertices, cache_num);
        FastSampler* cpu_sampler = new FastSampler(graph, fully_rep_graph,
                                                   cache_ids, 1, graph->gnnctx->fanout,
                                                   graph->config->batch_size);
        CPU_sg = cpu_sampler->sample_fast(cache_num);
        VertexId batch_start = 0;
        VertexId batch_end = std::min(cache_num, batch_start + cpu_batch_size);
        // CPU会按行进行聚合，所以行需要有序
        auto tmpX0 = ctx->runGraphOpNoBackward<nts::op::PushDownBatchOp>(CPU_sg, graph, 0,  F, batch_start, batch_end);
        auto* mask_ptr = new uint8_t[cache_num];
        NtsVar mask = torch::from_blob(mask_ptr, {cache_num, 1}, torch::kBool);
        mask.zero_();
//        NtsVar mask = torch::zeros({static_cast<long>(cache_ids.size()), 1}, torch::kBool);
//        auto* mask_ptr = mask.accessor<char , 2>().data();
        cpu_thread = std::thread([&](){
           while(!is_pipeline_end.load()){
               // 进行nn计算
               auto y = tmpX0.matmul(P[0]->W_c);
//               std::printf("tmp X0 sum: %lf\n", tmpX0.sum().item<double>());
               // 保存X0
               if(batch_start == 0) {
                   X0 = tmpX0;

               } else {
                   X0 = torch::cat({X0, tmpX0}, 0);
               }

               // 将结果传到gpu
               gnndatum->move_data_to_local_cache(y.size(0), y.size(1), y.accessor<ValueType , 2>().data(),
                                                  &(cache_ids_ptr[batch_start]), mask_ptr, batch_start);


               // 检查是否到最后，计算完了的话重启采样器，并进行采样, 然后进行反向
               if(batch_end == cache_num) {
                   cpu_sampler->restart();
                   CPU_sg = cpu_sampler->sample_fast(cache_num);
//                   std::printf("before mask X0 dim: %d\n", X0.dim());
//                   std::printf("before mask X0 size: (%d, %d)\n", X0.size(0), X0.size(1));
//                   std::printf("mask size: (%d, %d)\n", mask.size(0), mask.size(1));
                   auto col = X0.size(1);
                   X0 = torch::masked_select(X0, mask);
                   int64_t row = X0.size(0)/col;
                   assert(X0.size(0)%col == 0);
                   X0.resize_({row, col});
//                   std::printf("X0.size(%d, %d)\n", X0.size(0), X0.size(1));
//                   std::printf("after mask X0 dim: %d\n", X0.dim());
                   ExecCPUBackward(X0, mask); // 这里面也会重置一些变量
                   // 重置一些关键变量
                   mask.zero_();

               }
               batch_start = batch_end % cache_num;
               batch_end = std::min(cache_num, batch_start + cpu_batch_size);

               // 进行CPU的聚合
               tmpX0 = ctx->runGraphOpNoBackward<nts::op::PushDownBatchOp>(CPU_sg, graph, 0,  F, batch_start, batch_end);
           }
        });


        InitStream();

        int layer = graph->gnnctx->layer_size.size()-1;

        FastSampler* train_sampler = new FastSampler(fully_rep_graph,train_nids,layer,graph->gnnctx->fanout, pipeline_num, cuda_stream);
        FastSampler* eval_sampler = new FastSampler(fully_rep_graph,val_nids,layer,graph->gnnctx->fanout);
        FastSampler* test_sampler = new FastSampler(fully_rep_graph,test_nids,layer,graph->gnnctx->fanout);


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
        std::printf("update time: %lf (s)\n", update_time);
        std::printf("grad back time: %lf (s)\n", grad_back_time);
        std::printf("cal grad time: %lf (s)\n", cal_grad_time);
        std::printf("cal grad wait time: %lf (s)\n", cal_grad_wait_time);
        std::printf("update weight time: %lf (s)\n", update_weight_time);
        std::printf("cpu cal grad time: %lf (s)\n", cpu_cal_grad_time);
        std::printf("cpu reset flag time: %lf (s)\n", cpu_reset_flag_time);
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
        std::sort(cache_node_idx_seq.begin(), cache_node_idx_seq.end(), [&](const int x, const int y) {
            return graph->out_degree_for_backward[x] > graph->out_degree_for_backward[y];
        });

        for (int i = 1; i < graph->vertices; ++i) {
            assert(graph->out_degree_for_backward[cache_node_idx_seq[i]] <= graph->out_degree_for_backward[cache_node_idx_seq[i - 1]]);
        }

        for (int i = 0; i < graph->vertices; ++i) {
            gnndatum->CacheFlag[i] = -1; //init
        }

        int cache_node_num = graph->vertices * Cacherate;
        for (int i = 0; i < cache_node_num; ++i) {
            // LOG_DEBUG("cache_nodes[%d] = %d", i, cache_nodes[i]);
            gnndatum->CacheFlag[cache_node_idx_seq[i]] = 0; //初始化为cache顶点
        }
        cache_ids.resize(cache_node_num);
        std::copy(cache_node_idx_seq.begin(), cache_node_idx_seq.begin() + cache_node_num, cache_ids.begin());
    }

    inline void Forward(FastSampler* sampler, NtsVar& tmp_X0, SampledSubgraph* sg, Cuda_Stream* cudaStream){
        for(int l = 0; l < (graph->gnnctx->layer_size.size()-1); l++){//forward
            graph->rtminfo->curr_layer = l;
            int hop = (graph->gnnctx->layer_size.size()-2) - l;
            if(l == 0) {
                //
                NtsVar Y_i=ctx->runGraphOp<nts::op::SingleGPUAllSampleGraphOp>(sg,graph,hop,tmp_X0,cudaStream);
                X[l + 1] = ctx->runVertexForward([&](NtsVar n_i,NtsVar v_i){
                                                     return vertexForward(n_i, v_i);
                                                 },
                                                 Y_i,
                                                 tmp_X0);
                //load embedding of cacheflag = 2 or 3 to X[l + 1], GPU cache embedding
                sampler->load_share_embedding(cudaStream, sg, gnndatum->dev_share_embedding, X[l + 1],gnndatum->CacheMap,gnndatum->CacheFlag);
//                            P[0]->set_gradient_like(X[l+1]);
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
            //ctx->self_backward(false);
            ctx->self_backward_cache(false,
                                     gnndatum->CacheFlag,
                                     gnndatum->CacheMap,
                                     sg->sampled_sgs[graph->gnnctx->layer_size.size()-2]->dev_destination,
                                     sg->sampled_sgs[graph->gnnctx->layer_size.size()-2]->v_size,
                                     graph,
                                     cudaStream,
                                     P[0]->dev_w_gradient_buffer);
            //printf("#training_time_step = %lf(s)\n", training_time_step * 10);
            update_time -= get_time();
            if(pipeline_num == 1) {
                Update(true);
            } else if(batch % pipeline_num == pipeline_num - 1) {
                UpdateCache(true);
            } else {
                UpdateCache(false);
                if(batch % pipeline_num == pipeline_num - 2) {
                    start_send_flag = true;
                }
            }
            update_time += get_time();
            for (int i = 0; i < P.size(); i++) {
                if(i == 0){
                    P[i]->reset_layer();
                } else {
                    P[i]->zero_grad();
                }
            }
        }
    }

    inline void CheckFlagAndSendGrad(bool& start_send_flag){
        if(start_send_flag){
            SendGradToCpu();
            start_send_flag = false;
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

#endif //GNNMINI_GCN_SAMPLE_PD_CACHE_HPP
