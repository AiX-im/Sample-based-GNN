//
// Created by toao on 23-9-21.
//

#ifndef GNNMINI_GS_SAMPLE_PC_MULTI_HPP
#define GNNMINI_GS_SAMPLE_PC_MULTI_HPP
#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include <pthread.h>


class GS_SAMPLE_PC_MULTI_impl {
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
//    NtsVar MASK_gpu;
    std::vector<NtsVar> multi_MASK_gpu;
    //GraphOperation *gt;
    PartitionedGraph *partitioned_graph;
    std::vector<nts::ctx::NtsContext *> device_ctx;
//    nts::ctx::NtsContext *ctx;
    // Variables
//    std::vector<Parameter *> P;
    std::vector<std::vector<Parameter*>>  multi_P;
//    std::vector<NtsVar> X;
    std::vector<std::vector<NtsVar>> device_X;
//    Cuda_Stream* cuda_stream;
    std::vector<Cuda_Stream*> device_cuda_stream;
    int pipeline_num;
//    std::vector<at::cuda::CUDAStream> torch_stream;
    std::vector<std::vector<at::cuda::CUDAStream>> device_torch_stream;
    std::vector<FastSampler*> device_train_sampler;
    NCCL_Communicator* train_nccl_communicator;
    NCCL_Communicator* transfer_nccl_communicator;
    std::vector<unsigned long> corrects;
    std::vector<unsigned int> multi_batchsize;


    std::vector<VertexId> batch_cache_num;
    VertexId top_cache_num;
    VertexId super_batchsize;
    VertexId epoch_super_batch_num;
    VertexId last_super_batch_num;
    std::vector<std::vector<VertexId>> device_cache_offset;
    std::vector<VertexId> train_nids, val_nids, test_nids;
    float cache_rate;
    std::vector<VertexId> cache_ids;
    tbb::concurrent_queue<NtsVar> shared_W_queue;

    std::condition_variable cache_set_cv;
    std::mutex cache_set_mutex;
    std::vector<int> cache_set_batch;   // 用在多GPU之间确定该super_batch 运行到了第几个epoch

    // for feature cache
    std::vector<int> cache_node_idx_seq;
    VertexId* cache_node_hashmap;
    VertexId* dev_cache_node_hashmap;
    int cache_node_num = 0;
    float** dev_cache_feature;
    VertexId **local_idx, **local_idx_cache, **dev_local_idx, **dev_local_idx_cache;
    // VertexId *dev_cache_cnt, *local_cache_cnt;
    VertexId **outmost_vertex;
//    std::mutex sample_mutex;
//    std::mutex transfer_mutex;
//    std::mutex train_mutex;

//    std::vector<std::mutex> device_sample_mutex;
//    std::vector<std::mutex> device_transfer_mutex;
//    std::vector<std::mutex> device_train_mutex;


    NtsVar F;
//    NtsVar loss;
    NtsVar tt;
    torch::nn::Dropout drpmodel;
    FullyRepGraph* fully_rep_graph;
//    float acc;
//    int batch;
//    long correct;
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
    double update_cache_time = 0.0;
    double gpu_wait_time = 0.0;
    double cpu_wait_time = 0.0;

    int num_devices;

    GS_SAMPLE_PC_MULTI_impl(Graph<Empty> *graph_, int iterations_,
                           bool process_local = false,
                           bool process_overlap = false) {



        graph = graph_;
        iterations = iterations_;

        active = graph->alloc_vertex_subset();
        active->fill();

        // Note: 这里不涉及CUDA
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
        if(pipeline_num <= 0) {
            pipeline_num = 3;
        }
        wait_times = new double[pipeline_num];
        std::printf("pipeline num: %d\n", pipeline_num);



        LOG_INFO("config gpu num: %d", graph->config->gpu_num);
        num_devices = graph->config->gpu_num;
        if(num_devices == 0) {
            cudaGetDeviceCount(&num_devices);
        }
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
        train_nccl_communicator = new NCCL_Communicator(num_devices, arr.data(), 0);
        transfer_nccl_communicator = new NCCL_Communicator(num_devices, arr.data(), 0);
        corrects.resize(num_devices);
        multi_batchsize.resize(num_devices);
        graph->config->batch_size = graph->config->batch_size * num_devices;
    }
    void init_graph() {
        // Note: 这里不涉及CUDA
        fully_rep_graph=new FullyRepGraph(graph);
        fully_rep_graph->GenerateAll();
        fully_rep_graph->SyncAndLog("read_finish");

        // graph->init_message_buffer();
        // graph->init_communicatior();
        // NOTE: 这个涉及反向的栈，应该每个GPU都有一个
        for(int i = 0; i < num_devices; i++) {
            device_ctx[i] = new nts::ctx::NtsContext();
        }
//        ctx=new nts::ctx::NtsContext();
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

        // Note: 这里不涉及CUDA
        gnndatum = new GNNDatum(graph->gnnctx, graph);
        gnndatum->init_multi_gpu(num_devices, transfer_nccl_communicator);
        if (0 == graph->config->feature_file.compare("random")) {
            gnndatum->random_generate();
        } else if(0 == graph->config->feature_file.compare("mask")){
            gnndatum->read_mask_random_other(graph->config->mask_file);
        } else {
            gnndatum->readFeature_Label_Mask(graph->config->feature_file,
                                             graph->config->label_file,
                                             graph->config->mask_file);
        }
        // Note: 下面这两个函数都不涉及CUDA
        gnndatum->registLabel(L_GT_C);
        gnndatum->registMask(MASK);

        // TODO: 这里的零拷贝内存，可能会涉及CUDA，已修改，但可能零拷贝内存还会存在问题
//        gnndatum->genereate_gpu_data();
        gnndatum->generate_multi_gpu_data();
        // L_GT_G = L_GT_C.cuda();
        // NOTE： 这里涉及了CUDA，需要更改(已修改)
//        MASK_gpu = MASK.cuda();
        for(int i = 0; i < num_devices; i++) {
            multi_MASK_gpu[i] = MASK.to(torch::Device(torch::kCUDA, i));
        }

        // Note: 下面的都不涉及CUDA
//        for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
//            P.push_back(new Parameter(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i + 1], alpha, beta1, beta2, epsilon, weight_decay));
//            //            P.push_back(new Parameter(graph->gnnctx->layer_size[i],
//            //                        graph->gnnctx->layer_size[i+1]));
//        }
        for(int i = 0; i < num_devices; i++) {
            for(int j = 0; j < graph->gnnctx->layer_size.size() - 1; j++) {
                multi_P[i].push_back(new Parameter(graph->gnnctx->layer_size[j], graph->gnnctx->layer_size[j + 1], alpha, beta1, beta2, epsilon, weight_decay));
            }
        }

        // TODO: 下面的P会涉及CUDA，应该每个一个P
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

        // Note: 这个不涉及CUDA
        F = graph->Nts->NewLeafTensor(
                gnndatum->local_feature,
                {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
                torch::DeviceType::CPU);

        // Note: 这个只是初始化，也不涉及CUDA
        for(int device_id = 0; device_id < num_devices; device_id++) {
            for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
                NtsVar d;
                device_X[device_id].push_back(d);
            }

        }
        // X[0]=F.cuda().set_requires_grad(true);
        // NOTE: 这个好像实际也并没有使用
        for(int device_id = 0; device_id < num_devices; device_id++) {
            device_X[device_id][0] = F.set_requires_grad(true);
        }
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

    void Loss(NtsVar &left,NtsVar &right, nts::ctx::NtsContext* ctx) {
        //  return torch::nll_loss(a,L_GT_C);
        torch::Tensor a = left.log_softmax(1);
        //torch::Tensor mask_train = MASK_gpu.eq(0);
        auto loss = torch::nll_loss(a,right);
        if (ctx->training == true) {
            ctx->appendNNOp(left, loss);
        }
    }

    void Update(int device_id, cudaStream_t cudaStream) {
        for (int i = 0; i < multi_P[device_id].size(); i++) {
            // cudaStreamSynchronize(cudaStream);
            // TODO: 卡在了下面这里
//            std::printf("before all reduce device id: %d, layer: %d, stream: %p\n",
//                        device_id, i, cudaStream);
//            std::printf("grad sum: %.4f\n", multi_P[device_id][i]->W.grad().abs().sum().item<float>());

            multi_P[device_id][i]->reduce_multi_gpu_gradient( multi_P[device_id][i]->W.grad(),
                                                             device_id, cudaStream);

            // cudaStreamSynchronize(cudaStream);
//            std::printf("after all reduce device id: %d, layer: %d, stream: %p\n",
//                        device_id, i, cudaStream);
//            std::printf("device id: %d, layer %d before: %.4f\n", device_id, i, multi_P[device_id][i]->W.abs().sum().item<float>());
            // multi_P[device_id][i]->learnC2G_with_decay_Adam();

           multi_P[device_id][i]->learn_local_with_decay_Adam();
            // cudaDeviceSynchronize();
            multi_P[device_id][i]->next();
        }
    }

    NtsVar vertexForward(NtsVar &a, NtsVar &x, int device_id, int layer) {
        NtsVar y;
//        int layer = graph->rtminfo->curr_layer;
        int layer_num = gnndatum->gnnctx->layer_size.size() - 1;
        if (layer == layer_num - 1) {
//        if (layer == 1) {
            y = multi_P[device_id][layer]->forward(a);
            y = y.log_softmax(1); //CUDA

        } else {
            //y = P[layer]->forward(torch::relu(drpmodel(a)));
            auto b = multi_P[device_id][layer]->forward(a);
            y = torch::dropout(torch::relu(b), drop_rate, device_ctx[device_id]->is_train());
        }
        return y;
    }

    /**
     * @description: 执行前向计算
     * @param {FastSampler*} sampler 训练采样器
     * @param {int} type 0：train 1：eval 2：test
     * @return {*}
     */
    void Train(FastSampler* sampler, int epoch_num, int type=0, int device_id = 0) {
//        setPriority(1);
        // TODO: forward这里应该是每个GPU一个，所以这里的mutex，上锁这些应该是函数内变量
        graph->rtminfo->forward = true;
//        std::printf("device id: %d\n", device_id);
//        long correct = 0;
        corrects[device_id] = 0;
        device_ctx[device_id]->train();
        long batch = 0;
        // NtsVar target_lab;
        // X[0]=graph->Nts->NewLeafTensor({1000,F.size(1)},
        //   torch::DeviceType::CUDA);
        // target_lab=graph->Nts->NewLabelTensor({graph->config->batch_size},
        //       torch::DeviceType::CUDA);
        std::mutex sample_mutex;
        std::mutex transfer_mutex;
        std::mutex train_mutex;
        SampledSubgraph *sg[pipeline_num];
        NtsVar tmp_X0[pipeline_num];
        NtsVar tmp_target_lab[pipeline_num];
        for(int i = 0; i < pipeline_num; i++) {
            tmp_X0[i] = graph->Nts->NewLeafTensor({1000,F.size(1)},
                                                  torch::DeviceType::CUDA, device_id);
            tmp_target_lab[i] = graph->Nts->NewLabelTensor({graph->config->batch_size},
                                                           torch::DeviceType::CUDA, device_id);
        }
        std::thread threads[pipeline_num];
        std::vector<unsigned int> super_batch_countdown(epoch_super_batch_num); // 用于统计super batch内有多少batch已经完成
        // TODO: 最后一个super batch内batch数可能不等于pipeline数量，所以上面可以改为倒计数看看
        std::vector<bool>super_batch_ready(epoch_super_batch_num);

        std::printf("pipeline num: %d, batch size: %d\n", pipeline_num, multi_batchsize[device_id]);

        for(int tid = 0; tid < pipeline_num; tid++) {
            threads[tid] = std::thread([&](int thread_id){
                cudaSetUsingDevice(device_id);
//                LOG_DEBUG("device id: %d thread id: 0x%lx", device_id, std::this_thread::get_id());

                int local_batch;
                int super_batch_id;
                MultiCacheVars* cacheVars;
                NNVars* nnVars;

//                wait_times[thread_id] -= get_time();
                std::unique_lock<std::mutex> sample_lock(sample_mutex, std::defer_lock);
                sample_lock.lock();
//                wait_times[thread_id] += get_time();
//                std::printf("sample node num: %d\n", sampler->work_range[1]);
                while(sampler->sample_not_finished()){
//                    sample_time -= get_time();
                    local_batch = batch;
                    super_batch_id = local_batch / pipeline_num;
                    batch++;

                    // 如果是superbatch的第一个batch
                    // TODO: 这里涉及多个GPU使用同一个cache var来进行初始化，所以这里还需要进行修改
//                    std::printf("device id %d thread %d debug 1\n", device_id, thread_id);
                    if(local_batch % pipeline_num == 0) {
                        // 初始化cache
//                        std::printf("device id %d thread %d debug 1.1\n", device_id, thread_id);

                        cacheVars = gnndatum->multi_new_cache_var(super_batch_id, epoch_num);
                        cudaSetUsingDevice(device_id);
//                        std::printf("device id %d thread %d debug 1.2\n", device_id, thread_id);
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
//                        std::printf("device id %d thread %d debug 1.5\n", device_id, thread_id);

                        super_batch_countdown[super_batch_id] = 0;
                        super_batch_ready[super_batch_id] = false;
                    } else {
                        cacheVars = gnndatum->get_multi_cache_var(super_batch_id);
                    }

//                    std::printf("device id %d thread %d debug 2\n", device_id, thread_id);

                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
//                    sample_time -= get_time();
                    write_add(&sample_time, -get_time());
                    // TODO: 这里可能
                    sg[thread_id]=sampler->sample_gpu_fast_omit(multi_batchsize[device_id], thread_id,
                                            cacheVars->multi_dev_cache_map[device_id], super_batch_id);
                    //sg=sampler->sample_fast(graph->config->batch_size);
                    assert(sg[thread_id]->cs == &device_cuda_stream[device_id][thread_id]);
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
//                    cudaStreamSynchronize(device_cuda_stream[device_id][thread_id].stream);
                    write_add(&sample_time, get_time());
//                    std::printf("sample time: %.4lf\n", sample_time);
//                    sample_time += get_time();
                    sample_lock.unlock();
//                    std::printf("device id %d thread %d debug 3\n", device_id, thread_id);

                    std::unique_lock<std::mutex> transfer_lock(transfer_mutex, std::defer_lock);
                    transfer_lock.lock();
//                    std::printf("device id %d thread %d debug 4\n", device_id, thread_id);
                    write_add(&transfer_feature_time, -get_time());
//                    transfer_feature_time -= get_time();
                    // sampler->load_label_gpu(target_lab,gnndatum->dev_local_label);
                    // sampler->load_feature_gpu(X[0],gnndatum->dev_local_feature);
                    sampler->load_label_gpu(&device_cuda_stream[device_id][thread_id], sg[thread_id],
                                            tmp_target_lab[thread_id],gnndatum->dev_local_label_multi[device_id]);
                    // std::printf("device id %d thread %d debug 5\n", device_id, thread_id);
                    // sampler->load_feature_gpu(&device_cuda_stream[device_id][thread_id], sg[thread_id],
                    //                           tmp_X0[thread_id],gnndatum->dev_local_feature_multi[device_id]);

                    if (!graph->config->cacheflag) {  // trans feature use zero copy (omit gather feature)
                        sampler->load_feature_gpu(&device_cuda_stream[device_id][thread_id], sg[thread_id],
                         tmp_X0[thread_id],gnndatum->dev_local_feature_multi[device_id]);                        
                        // get_gpu_mem(used_gpu_mem, total_gpu_mem);
                    } else {  // trans freature which is not cache in gpu
                        // epoch_transfer_feat_time -= get_time();
                            sampler->load_feature_gpu_cache(
                            &device_cuda_stream[device_id][thread_id], sg[thread_id], tmp_X0[thread_id], gnndatum->dev_local_feature_multi[device_id],
                            dev_cache_feature[device_id], local_idx[device_id], local_idx_cache[device_id], cache_node_hashmap, dev_local_idx[device_id],
                            dev_local_idx_cache[device_id], dev_cache_node_hashmap, outmost_vertex[device_id]);
                    }
                    // std::printf("device id %d thread %d debug 6\n", device_id, thread_id);
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
//                    std::printf("device id %d thread %d debug 7\n", device_id, thread_id);
                    write_add(&transfer_feature_time, get_time());
//                    transfer_feature_time += get_time();

                    super_batch_countdown[super_batch_id]++;
                    if(super_batch_id != epoch_super_batch_num - 1) {
                        if(super_batch_countdown[super_batch_id] == pipeline_num) {
                            super_batch_ready[super_batch_id] = true;
                        }
                    } else if(super_batch_countdown[super_batch_id] == last_super_batch_num) {
                        super_batch_ready[super_batch_id] = true;
                    }
//                    std::printf("device id %d thread %d debug 8\n", device_id, thread_id);
                    transfer_lock.unlock();

                    while(!super_batch_ready[super_batch_id]) {
                        std::this_thread::yield();
                    }

                    std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
                    train_lock.lock();
                    double training_time_step = 0;
//                    training_time -= get_time();
                    write_add(&training_time, -get_time());
                    assert(sg[thread_id]->cs == &device_cuda_stream[device_id][thread_id]);
//                    std::printf("device id %d thread %d debug 9\n", device_id, thread_id);
//                    gpu_wait_time -= get_time();
                    write_add(&gpu_wait_time, -get_time());
                    if(super_batch_countdown[super_batch_id] == pipeline_num  || (super_batch_id == epoch_super_batch_num - 1&&
                                                                                  super_batch_countdown[super_batch_id] == last_super_batch_num )) {
                        nnVars = gnndatum->multi_new_nn_var(super_batch_id, device_id);
//                        gnndatum->move_nn_data_to_gpu(nnVars, cuda_stream[thread_id].stream);
                    } else {
                        nnVars = gnndatum->get_multi_nn_var(super_batch_id, device_id);
                    }
//                    gpu_wait_time += get_time();
                    write_add(&gpu_wait_time, get_time());

//                    std::printf("device id %d thread %d debug 10\n", device_id, thread_id);
                    at::cuda::setCurrentCUDAStream(device_torch_stream[device_id][thread_id]);
                    Forward(sampler, tmp_X0[thread_id], sg[thread_id], &device_cuda_stream[device_id][thread_id],
                            super_batch_id, cacheVars, nnVars, device_id);

                    // device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();

//                    std::printf("device id %d thread %d debug 11\n", device_id, thread_id);
                    Loss(device_X[device_id][graph->gnnctx->layer_size.size()-1],
                         tmp_target_lab[thread_id], device_ctx[device_id]);

//                    std::printf("device id %d thread %d debug 12\n", device_id, thread_id);
                    if (device_ctx[device_id]->training) {
                        training_time_step -= get_time();
                        // TODO: 有个线程卡在了里面
//                        std::printf("batch %d before backward device id: %d thread id: %d 0x%lx\n",
//                                    local_batch, device_id, thread_id, std::this_thread::get_id());
                        device_ctx[device_id]->self_backward(false);
//                        std::printf("batch %d after backward device id: %d thread id: %d 0x%lx\n",
//                                    local_batch, device_id, thread_id, std::this_thread::get_id());

                        training_time_step += get_time();
                        //printf("#training_time_step = %lf(s)\n", training_time_step * 10);
                        Update(device_id, device_cuda_stream[device_id][thread_id].stream);
                        for (int i = 0; i < multi_P[device_id].size(); i++) {
                            multi_P[device_id][i]->zero_grad();
                        }
                    }
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
//                    cudaStreamSynchronize(device_cuda_stream[device_id][thread_id].stream);
//                    std::printf("device id %d thread %d debug 13\n", device_id, thread_id);
                    write_add(&training_time, get_time());
//                    training_time += get_time();

                    corrects[device_id] += getCorrect(device_X[device_id][graph->gnnctx->layer_size.size()-1],
                                          tmp_target_lab[thread_id]);


                    if((super_batch_id != epoch_super_batch_num - 1 && super_batch_countdown[super_batch_id] == pipeline_num)
                       || (super_batch_id == epoch_super_batch_num - 1 && super_batch_countdown[super_batch_id] == last_super_batch_num)) {
                        if(device_id == 0){
                            shared_W_queue.push(multi_P[device_id][0]->W.cpu());
                        }
                    }
//                    std::printf("device id %d thread %d debug 14\n", device_id, thread_id);
                    super_batch_countdown[super_batch_id]--;
                    if(super_batch_countdown[super_batch_id] == 0) {
                        // 最后一个batch了，回收相应的空间
                        gnndatum->recycle_multi_gpu_memory(cacheVars, nnVars, super_batch_id, device_id);
                    }

//                    std::printf("device id %d thread %d debug 15\n", device_id, thread_id);
                    train_lock.unlock();
//                    std::printf("correct num: %ld\n", corrects[device_id]);
                    sample_lock.lock();

                }
                sample_lock.unlock();
//                LOG_DEBUG("device id: %d thread id: 0x%lx finished", device_id, std::this_thread::get_id());

            }, tid);
        }
        for(int i = 0; i < pipeline_num; i++) {
            threads[i].join();
        }
        sampler->restart();
//        float acc = 1.0 * correct / sampler->work_range[1];
//        if (type == 0) {
//            LOG_INFO("Train Acc: %f %d %d", acc, correct, sampler->work_range[1]);
//        } else if (type == 1) {
//            LOG_INFO("Eval Acc: %f %d %d", acc, correct, sampler->work_range[1]);
//        } else if (type == 2) {
//            LOG_INFO("Test Acc: %f %d %d", acc, correct, sampler->work_range[1]);
//        }
        // loss=X[graph->gnnctx->layer_size.size()-1];
        // #run_time=1784.571165(s)
        // exec_time=1842.431756(s) reddit

        // run_time=2947.184987(s) cpu
        // exec_time=2986.859283(s)
    }

    void run() {
        if (graph->partition_id == 0)
            printf("GNNmini::Engine[Dist.GPU.GSimpl] running [%d] Epochs\n",
                   iterations);

        //      graph->print_info();
        // Note：该部分不涉及CUDA
//        std::vector<VertexId> train_nids, val_nids, test_nids;
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
        // TODO: 从文件中读取缓存顶点时应该是在shuffle前的，所以local shuffle时应该返回supper_batch_id，这样就可以从文件中找到需要的cache顶点了

//        nts::op::nts_local_shuffle(train_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);
//        nts::op::nts_local_shuffle(val_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);
//        nts::op::nts_local_shuffle(test_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);

        std::vector<int> fanout(1);
        fanout[0] = graph->gnnctx->fanout[graph->gnnctx->fanout.size() - 1];
        initCacheVariable();
        determine_cache_node_idx(graph->vertices * graph->config->feature_cache_rate);
        std::printf("cache ids size: %d\n", cache_ids.size());
        nts::op::nts_local_shuffle(train_nids,  graph->config->batch_size * pipeline_num, cache_ids, batch_cache_num);
        initDeviceOffset();
        // TODO: 下面的cache_ids为空了
        FastSampler* cpu_sampler = new FastSampler(graph, fully_rep_graph,
                                                   cache_ids, 1, fanout,
                                                   super_batchsize);

        InitCudaStream();

//        int layer = graph->gnnctx->layer_size.size()-1;
//        if(graph->config->pushdown)
//            layer--;
        // TODO: 这里涉及了CUDA，以及多流，可以考虑每个GPU一个采样器
        // 初始化多个采样器
        initMultiSampler();

        // 因为保持了每个super batch，所以不同device之间也是共用这个cache标记即可
        // 因为cache标志对应的是super batch的
//        gnndatum->init_super_batch_var(epoch_super_batch_num);
        gnndatum->init_multi_cache_var(epoch_super_batch_num, num_devices);
        LOG_INFO("Finished init_multi_cache_var");

        shared_W_queue.push(multi_P[0][0]->W.cpu());
        auto total_super_batch_num = epoch_super_batch_num * iterations;

        // 为每个GPU都创建一个流
        cudaStream_t cpu_cuda_streams[num_devices];
        for(int i = 0; i < num_devices; i++) {
            cudaSetUsingDevice(i);
            cudaStreamCreateWithFlags(&cpu_cuda_streams[i], cudaStreamNonBlocking);
        }
        LOG_INFO("start cpu_sampler->sample_fast");
        SampledSubgraph* CPU_sg = cpu_sampler->sample_fast(batch_cache_num[0], WeightType::Sum);
//        VertexId batch_start = 0;
//        VertexId batch_end = std::min((VertexId)cache_ids.size(), batch_start + batch_cache_num);
        LOG_INFO("start CPU_Forward");
        NtsVar embedding = CPU_Forward(CPU_sg);
        std::printf("batch_cache_num: %d, device_cache_offset: %d, sg dst: %d\n", batch_cache_num[0],
                    device_cache_offset[0][num_devices], CPU_sg->sampled_sgs[0]->dst().size());

        gnndatum->move_embedding_to_multi_gpu(top_cache_num, embedding.size(0), embedding.size(1),
                                        embedding.accessor<ValueType, 2>().data(),
                                        cpu_cuda_streams, num_devices, device_cache_offset[0]);

        auto cpu_thread = std::thread([&]() {
//            setPriority(99);
            for(VertexId i = 1; i < total_super_batch_num; i++) {
                if(i % epoch_super_batch_num == 0) {
                    cpu_sampler->restart();
                }
//                batch_start = batch_end % cache_ids.size();
//                batch_end = std::min(batch_start + batch_cache_num, (VertexId)cache_ids.size());
                CPU_sg = cpu_sampler->sample_fast(batch_cache_num[i % epoch_super_batch_num], WeightType::Sum);
                embedding = CPU_Forward(CPU_sg);

                gnndatum->move_embedding_to_multi_gpu(top_cache_num, embedding.size(0), embedding.size(1),
                                                      embedding.accessor<ValueType, 2>().data(),
                                                      cpu_cuda_streams, num_devices,
                                                      device_cache_offset[i % epoch_super_batch_num]);
//                assert(batch_cache_num[i % epoch_super_batch_num] == device_cache_offset[i % epoch_super_batch_num][num_devices]);
            }
        });


//        FastSampler* train_sampler = new FastSampler(fully_rep_graph,train_nids,layer,graph->gnnctx->fanout, pipeline_num, device_cuda_stream[0]);
//        FastSampler* eval_sampler = new FastSampler(fully_rep_graph,val_nids,layer,graph->gnnctx->fanout);
//        FastSampler* test_sampler = new FastSampler(fully_rep_graph,test_nids,layer,graph->gnnctx->fanout);

        // FastSampler* train_sampler = new FastSampler(graph,fully_rep_graph,
        //     train_nids,layer,
        //         graph->gnnctx->fanout,graph->config->batch_size,true);

        // FastSampler* eval_sampler = new FastSampler(graph,fully_rep_graph,
        //     val_nids,layer,
        //         graph->gnnctx->fanout,graph->config->batch_size,true);

        // FastSampler* test_sampler = new FastSampler(graph,fully_rep_graph,
        //     test_nids,layer,
        //         graph->gnnctx->fanout,graph->config->batch_size,true);
        auto total_node_num = train_nids.size();
        auto start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        exec_time -= get_time();
        for (int i_i = 0; i_i < iterations; i_i++) {
            double per_epoch_time = 0.0;
            per_epoch_time -= get_time();
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
//            ctx->train();

            std::vector<std::thread> multi_thread;
            for(int device_id = 0; device_id < num_devices; device_id++) {
                multi_thread.emplace_back([&](int dev_id){
                    Train(device_train_sampler[dev_id], i_i, 0, dev_id);
                }, device_id);
                setScheduling(multi_thread[device_id], SCHED_RR, 1);
            }
            for(int device_id = 0; device_id < num_devices; device_id++) {
                multi_thread[device_id].join();
            }

            TestMiniBatchAll(0, total_node_num);
            // ctx->eval();
            // Forward(eval_sampler, 1);
            // Forward(test_sampler, 2);
            per_epoch_time += get_time();

            std::cout << "GNNmini::Running.Epoch[" << i_i << "]:Times["
                      << per_epoch_time << "(s)]" << std::endl;
        }
        auto end_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        exec_time += get_time();
        cpu_thread.join();
        auto wait_time = 0.0;
        for(int i = 0; i < pipeline_num; i++) {
            wait_time += wait_times[i];
        }
        printf("#run_time=%lf(s)\n", exec_time);
//        printf("all:%lf(s) prepro:%lf(s) pro:%lf(s) post:%lf(s) copy:%lf(s)\n",train_sampler->all_time,train_sampler->pre_pro_time, train_sampler->pro_time,train_sampler->post_pro_time,train_sampler->copy_gpu_time );
//        printf("test_time:%lf(s)\n",train_sampler->test_time);
        printf("#wait time: %lf(s)\n", wait_time);
        printf("#gather_feature_time=%lf(s)\n", gather_feature_time);
        for(auto&& train_sampler: device_train_sampler){
            std::printf("cpu inclusiveTime: %.4lf (s)\n", train_sampler->cs->cpu_inclusiveTime);
            std::printf("inclusiveTime: %.4lf (s)\n", train_sampler->cs->inclusiveTime);
            std::printf("init layer time: %.4lf (s)\n", train_sampler->init_layer_time);
            std::printf("init co time: %.4lf (s)\n", train_sampler->init_co_time);
            std::printf("pro time: %.4lf (s)\n", train_sampler->pro_time);
            std::printf("post pro time: %.4lf (s)\n", train_sampler->post_pro_time);
        }
        printf("#sample_time= %.4lf (s)\n", (sample_time));
        printf("#transfer_feature_time= %.4lf (s)\n", (transfer_feature_time));
        printf("#training_time= %.4lf (s)\n", training_time);
        delete active;
        printf("#average epoch time: %.4lf (s)\n", exec_time/iterations);
        std::printf("#cpu wait time: %.4lf (s)\n", cpu_wait_time);
        std::printf("#gpu wait time: %.4lf (s)\n", gpu_wait_time);
        printf("总传输节点数: %llu\n", Cuda_Stream::total_transfer_node);
//        printf("平均epoch传输节点数:%llu\n", Cuda_Stream::total_transfer_node/iterations);
        printf("%lu\n%lu\n", start_time, end_time);
    }

    NtsVar CPU_Forward(SampledSubgraph* CPU_sg) {
        auto tmpX0 = device_ctx[0]->runGraphOpNoBackward<nts::op::CPUPushDownOp>(CPU_sg, graph,
                                               0,  F);

        NtsVar W;
        cpu_wait_time -= get_time();
        while(!shared_W_queue.try_pop(W)){
            std::this_thread::yield();
        }
        cpu_wait_time += get_time();

        auto y = tmpX0.matmul(W);
        return y;
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
        cache_ids = nts::op::preSample(train_nids, graph->config->batch_size, batch_cache_num, graph->config->cache_rate, top_cache_num,
                                       gnndatum->gnnctx->layer_size.size() - 1, fully_rep_graph,
                                       graph->config->cache_rate / 0.8, graph,  pipeline_num);
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

//        for(int i = 0; i < epoch_super_batch_num; i++) {
//            std::printf("i: %d, offset: %d, batch cache: %d\n", i,
//                        device_cache_offset[i][num_devices], batch_cache_num[i]);
//        }
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
        dev_cache_feature = new float *[num_devices];
        for(int i = 0; i < num_devices; i++) {
            cudaSetUsingDevice(i);
            dev_cache_feature[i] = (float*)cudaMallocGPU(cache_node_num * sizeof(float) * feat_dim);
        }
        // gather_cache_feature, prepare trans to gpu
        LOG_DEBUG("start gather_cpu_cache_feature");
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
        local_idx = new VertexId *[num_devices];
        local_idx_cache = new VertexId *[num_devices];
        dev_local_idx = new VertexId *[num_devices];
        dev_local_idx_cache = new VertexId *[num_devices];
        outmost_vertex = new VertexId *[num_devices];

        LOG_DEBUG("start trans to gpu");
        for(int i = 0; i < num_devices; i++) {
            cudaSetUsingDevice(i);
            move_data_in(dev_cache_feature[i], local_cache_feature_gather, 0, cache_node_num, feat_dim);
            local_idx[i] = (VertexId*)cudaMallocPinned(1ll * graph->vertices * sizeof(VertexId));
            local_idx_cache[i] = (VertexId*)cudaMallocPinned(1ll * graph->vertices * sizeof(VertexId));
            outmost_vertex[i] = (VertexId*)malloc(graph->vertices * sizeof(VertexId));
            dev_local_idx[i] = (VertexId*)getDevicePointer(local_idx[i]);
            dev_local_idx_cache[i] = (VertexId*)getDevicePointer(local_idx_cache[i]);
        }
    }


    inline void initDeviceOffset(){

        // 为每个GPU初始化传输偏移量
        device_cache_offset.resize(epoch_super_batch_num);
        for(int i = 0; i < epoch_super_batch_num; i++) {
            device_cache_offset[i].resize(num_devices+1);
            device_cache_offset[i][0] = 0;
            size_t per_device = batch_cache_num[i] / num_devices;
            for(int j = 1; j < num_devices; j++) {
                device_cache_offset[i][j] = per_device * j;
            }
            device_cache_offset[i][num_devices] = batch_cache_num[i];
        }
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

        // 最后一个batch的大小
        last_batchsize = train_nids.size() % total_batchsize;
        if(last_batchsize != 0 && last_batchsize < num_devices) {
            for(int i = last_batchsize; i < num_devices; i++) {
                train_nids.push_back(0);
            }
            last_batchsize = num_devices;
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
        LOG_INFO("out of loop");
        std::printf("train_nids.size(): %lu, total_batchsize: %u, last_batchsize: %u, num_device: %d\n",
                    train_nids.size(), total_batchsize, last_batchsize, num_devices);

        
        uint32_t  last_per_device = last_batchsize / num_devices;
        uint32_t  last_per_device_end = 0;
        if(last_per_device != 0) {
            last_per_device_end = last_batchsize % (last_per_device * num_devices) + last_per_device;
        }

        std::printf("start copy ids to multi device array\n");
        // 复制顶点id到相应数组
        start = 0;
        for(uint32_t i = 0; i < total_batch_num; i++) {
            // 如果是最后一个batch且相除不为零
            if(i == total_batch_num - 1 && last_batchsize != 0) {
                if(last_per_device == 0){
                    break;
                }
                std::printf("start: %u, last_per_device: %u, num_devices: %u, last_batchsize: %u, total_batchsize: %u, last_per_device_end: %u\n",
                            start, last_per_device, num_devices, last_batchsize, total_batchsize, last_per_device_end);
                std::printf("前者: %u, 后者: %u\n", start + last_per_device * num_devices + last_batchsize % (last_per_device*num_devices), train_nids.size());
                assert(start + last_per_device * num_devices + last_batchsize %  (last_per_device*num_devices) == train_nids.size());
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
        std::printf("start new device sampler\n");

        for(int i = 0; i < num_devices; i++) {
            cudaSetUsingDevice(i);
            std::printf("train id size: %d\n", multi_train_nids[i].size());
            device_train_sampler[i] = new FastSampler(fully_rep_graph,multi_train_nids[i],layer, graph->config->batch_size,
                                                      graph->gnnctx->fanout, pipeline_num, device_cuda_stream[i]);                     
            std::printf("sampler train id size: %d\n", device_train_sampler[i]->work_range[1]);
        }
        std::printf("finish new device sampler\n");
    }

    void InitCudaStream() {

        // TODO: 这里涉及了CUDA，应该为每个流创建一个CUDA
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
    inline void Forward(FastSampler* sampler, NtsVar& tmp_X0, SampledSubgraph* sg, Cuda_Stream* cudaStream,
                        VertexId super_batch_id, MultiCacheVars* cacheVars, NNVars* nnVars, int device_id){
//         print_tensor_size(tmp_X0);
        for(int l = 0; l < (graph->gnnctx->layer_size.size()-1); l++){//forward
            graph->rtminfo->curr_layer = l;
            int hop = (graph->gnnctx->layer_size.size()-2) - l;
            if(l == 0) {
                // Note: 下面用于debug
//                sampler->print_avg_weight(cudaStream, sg, gnndatum->dev_CacheFlag);
                assert(cudaStream == sg->cs);
                NtsVar Y_i=device_ctx[device_id]->runGraphOp<nts::op::SingleGPUAllSampleGraphOp>(sg,
                                             graph,hop,tmp_X0,cudaStream,
                                             device_id);;
//                sampler->load_share_aggregate(cudaStream, sg,gnndatum->dev_share_aggregate,
//                                                          Y_i, gnndatum->CacheMap,gnndatum->CacheFlag);

                device_X[device_id][l + 1] = device_ctx[device_id]->runVertexForward(
                        [&](NtsVar n_i,NtsVar v_i){
                            auto Y_W = MultiplyWeight(n_i, l, device_id);
                            // cudaStream->CUDA_DEVICE_SYNCHRONIZE();
//                            update_cache_time -= get_time();
                            write_add(&update_cache_time, -get_time());
                            sampler->load_share_embedding(cudaStream, sg, nnVars->dev_shared_embedding,
                                                          Y_W, cacheVars->multi_dev_cache_map[device_id],
                                                          cacheVars->multi_dev_cache_location[device_id],
                                                          super_batch_id);
                            // cudaStream->CUDA_DEVICE_SYNCHRONIZE();
//                            update_cache_time += get_time();
                            write_add(&update_cache_time, get_time());
//                                         Y_sum = Y_i.abs().sum().item<double>();
//                                        std::printf("after Y row: %d, sum: %lf, avg: %lf\n", Y_i.size(0), Y_sum, Y_sum/Y_i.size(0));
                            return RunReluAndDropout(Y_W, device_id);
                        },
                        Y_i,
                        tmp_X0
                );
            } else {
                NtsVar Y_i = device_ctx[device_id]->runGraphOp<nts::op::SingleGPUAllSampleGraphOp>(sg,graph,
                                                   hop,device_X[device_id][l],cudaStream, device_id);
                device_X[device_id][l + 1] = device_ctx[device_id]->runVertexForward([&](NtsVar n_i,NtsVar v_i){
                                                     return vertexForward(n_i, v_i, device_id, l);
                                                 },
                                                 Y_i,
                                               device_X[device_id][l]);
            }
        }
    }

    NtsVar MultiplyWeight(NtsVar& a, int layer, int device_id){
        return multi_P[device_id][layer]->forward(a);
    }
    NtsVar RunReluAndDropout(NtsVar& a, int device_id) {
        return torch::dropout(torch::relu(a), drop_rate, device_ctx[device_id]->is_train());
    }

    void setScheduling(std::thread &th, int policy, int priority) {
        sched_param sch_params;
        sch_params.sched_priority = priority;
        if(pthread_setschedparam(th.native_handle(), policy, &(sch_params))) {
            std::cerr << "Failed to set Thread scheduling : " << std::strerror(errno) << std::endl;
        }
    }

    void setPriority(int priority) {
        int policy = SCHED_FIFO; // 设置为FIFO调度策略
        struct sched_param param;
        param.sched_priority = priority; // 设置线程优先级

        int result = sched_setscheduler(0, policy, &param);
        if (result == -1) {
            std::cerr << "Failed to set thread priority." << std::endl;
            return;
        }

        // 在这里执行线程工作
        // std::this_thread::sleep_for(std::chrono::seconds(5));
    }

};

#endif //GNNMINI_GS_SAMPLE_PC_MULTI_HPP
