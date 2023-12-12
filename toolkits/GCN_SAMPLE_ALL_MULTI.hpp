//
// Created by toao on 23-9-21.
//

#ifndef GNNMINI_GCN_SAMPLE_ALL_MULTI_HPP
#define GNNMINI_GCN_SAMPLE_ALL_MULTI_HPP
#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
class GCN_SAMPLE_ALL_MULTI_impl {
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
    std::vector<unsigned long> corrects;


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

    int num_devices;

    void initMultiDeviceVar() {
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
        train_nccl_communicator = new NCCL_Communicator(num_devices, arr.data());
//        NCCL_Communicator::initAllNCCLComm(num_devices, arr.data());
//        for(int i = 0; i < num_devices; i++){
//            nccl_communicators.push_back(new NCCL_Communicator(i, 0));
//        }
        corrects.resize(num_devices);
        graph->config->batch_size = graph->config->batch_size * num_devices;
    }

    GCN_SAMPLE_ALL_MULTI_impl(Graph<Empty> *graph_, int iterations_,
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
        if(pipeline_num <= 0) {
            pipeline_num = 3;
        }
        wait_times = new double[pipeline_num];
        std::printf("pipeline num: %d\n", pipeline_num);

        initMultiDeviceVar();
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
        gnndatum->init_multi_gpu(num_devices, train_nccl_communicator);
        if (0 == graph->config->feature_file.compare("random")) {
            gnndatum->random_generate();
        } else if(0 == graph->config->feature_file.compare("mask")){
            gnndatum->read_mask_random_other(graph->config->mask_file);
        }  else {
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

        // 下面使用多线程进行初始化
//        std::vector<std::thread> init_threads;
//        for(int i = 0; i < num_devices; i++) {
//            init_threads.emplace_back([&](int dev_id){
//                torch::Device GPU(torch::kCUDA, dev_id);
//                for(int j = 0; j < multi_P[dev_id].size(); j++) {
//                    std::printf("device_id: %d, nccl dev id: %d\n", dev_id, nccl_communicators[dev_id]->getDeviceId());
//                    multi_P[dev_id][j]->set_multi_gpu_comm(nccl_communicators[dev_id]);
//                    multi_P[dev_id][j]->init_multi_gpu_parameter();
//                    multi_P[dev_id][j]->set_decay(decay_rate, decay_epoch);
//                    multi_P[dev_id][j]->to(GPU);
//                    multi_P[dev_id][j]->Adam_to_GPU(dev_id);
//                    cudaDeviceSynchronize();
//                }
//            }, i);
//        }
//        for(int i = 0; i < num_devices; i++) {
//            init_threads[i].join();
//        }

        //        F=graph->Nts->NewOnesTensor({graph->gnnctx->l_v_num,
        //        graph->gnnctx->layer_size[0]},torch::DeviceType::CPU);

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
            multi_P[device_id][i]->reduce_multi_gpu_gradient(multi_P[device_id][i]->W.grad(),
                                                             device_id, cudaStream);
//            std::printf("device id: %d, before: %.4f\n", device_id, multi_P[device_id][i]->W.abs().sum().item<float>());
            multi_P[device_id][i]->learn_local_with_decay_Adam();
//            cudaDeviceSynchronize();
//            std::printf("device id: %d, after: %.4f\n", device_id, multi_P[device_id][i]->W.abs().sum().item<float>());
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
    void Forward(FastSampler* sampler, int type=0, int device_id = 0) {
        // TODO: forward这里应该是每个GPU一个，所以这里的mutex，上锁这些应该是函数内变量
        graph->rtminfo->forward = true;
//        std::printf("device id: %d\n", device_id);
//        long correct = 0;
        corrects[device_id] = 0;
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
        for(int tid = 0; tid < pipeline_num; tid++) {
            threads[tid] = std::thread([&](int thread_id){
                cudaSetUsingDevice(device_id);
//                wait_times[thread_id] -= get_time();
                std::unique_lock<std::mutex> sample_lock(sample_mutex, std::defer_lock);
                sample_lock.lock();
//                wait_times[thread_id] += get_time();
//                std::printf("sample node num: %d\n", sampler->work_range[1]);
                while(sampler->sample_not_finished()){
//                    sample_time -= get_time();
                    write_add(&sample_time, -get_time());
                    sg[thread_id]=sampler->sample_gpu_fast(graph->config->batch_size/num_devices, thread_id);
                    //sg=sampler->sample_fast(graph->config->batch_size);
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
//                    cudaStreamSynchronize(device_cuda_stream[device_id][thread_id].stream);
                    write_add(&sample_time, get_time());
                    sample_lock.unlock();

                    std::unique_lock<std::mutex> transfer_lock(transfer_mutex, std::defer_lock);
                    transfer_lock.lock();
                    write_add(&transfer_feature_time, -get_time());
//                    transfer_feature_time -= get_time();
                    // sampler->load_label_gpu(target_lab,gnndatum->dev_local_label);
                    // sampler->load_feature_gpu(X[0],gnndatum->dev_local_feature);
                    sampler->load_label_gpu(&device_cuda_stream[device_id][thread_id], sg[thread_id],
                                            tmp_target_lab[thread_id],gnndatum->dev_local_label_multi[device_id]);
                    sampler->load_feature_gpu(&device_cuda_stream[device_id][thread_id], sg[thread_id],
                                              tmp_X0[thread_id],gnndatum->dev_local_feature_multi[device_id]);
                    device_cuda_stream[device_id][thread_id].CUDA_DEVICE_SYNCHRONIZE();
                    write_add(&transfer_feature_time, get_time());
//                    transfer_feature_time += get_time();
                    transfer_lock.unlock();

                    std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
                    train_lock.lock();
                    double training_time_step = 0;
//                    training_time -= get_time();
                    write_add(&training_time, -get_time());


                    at::cuda::setCurrentCUDAStream(device_torch_stream[device_id][thread_id]);
                    // std::printf("after setCurrentCUDAStream\n");
                    for(int l = 0; l < (graph->gnnctx->layer_size.size()-1); l++){//forward
                        graph->rtminfo->curr_layer = l;
                        int hop = (graph->gnnctx->layer_size.size()-2) - l;
                        if(l == 0) {
                            assert(sg[thread_id]->cs == &device_cuda_stream[device_id][thread_id]);
                            NtsVar Y_i=device_ctx[device_id]->runGraphOp<nts::op::SingleGPUAllSampleGraphOp>(sg[thread_id],
                                                                        graph,hop,tmp_X0[thread_id],&device_cuda_stream[device_id][thread_id],
                                                                        device_id);
                            device_X[device_id][l + 1] = device_ctx[device_id]->runVertexForward([&](NtsVar n_i,NtsVar v_i){
                                                                 return vertexForward(n_i, v_i, device_id,
                                                                                      l);
                                                             },
                                                             Y_i,
                                                             tmp_X0[thread_id]);
//                            cudaDeviceSynchronize();
//                            auto Y1_sum = Y_i.abs().sum().item<double>();
//                            std::printf("X1数量: %d, X1 sum: %lf, X1 avg: %lf\n", Y_i.size(0), Y1_sum, Y1_sum/Y_i.size(0));
                        } else {
                            NtsVar Y_i = device_ctx[device_id]->runGraphOp<nts::op::SingleGPUAllSampleGraphOp>(sg[thread_id],
                                                                           graph,hop,device_X[device_id][l],&device_cuda_stream[device_id][thread_id],
                                                                           device_id);
                            device_X[device_id][l + 1] = device_ctx[device_id]->runVertexForward([&](NtsVar n_i,NtsVar v_i){
                                                                 return vertexForward(n_i, v_i, device_id, l);
                                                             },
                                                             Y_i,
                                                             device_X[device_id][l]);
                        }
                    }

                    Loss(device_X[device_id][graph->gnnctx->layer_size.size()-1],
                         tmp_target_lab[thread_id], device_ctx[device_id]);
                    if (device_ctx[device_id]->training) {
                        training_time_step -= get_time();
                        device_ctx[device_id]->self_backward(false);
                        training_time_step += get_time();
                        //printf("#training_time_step = %lf(s)\n", training_time_step * 10);
                        Update(device_id, device_cuda_stream[device_id][thread_id].stream);
                        for (int i = 0; i < multi_P[device_id].size(); i++) {
                            multi_P[device_id][i]->zero_grad();
                        }
                    }
                    cudaStreamSynchronize(device_cuda_stream[device_id][thread_id].stream);
                    write_add(&training_time, get_time());
//                    training_time += get_time();

                    corrects[device_id] += getCorrect(device_X[device_id][graph->gnnctx->layer_size.size()-1],
                                          tmp_target_lab[thread_id]);
                    batch++;

                    train_lock.unlock();
//                    std::printf("correct num: %ld\n", corrects[device_id]);
                    sample_lock.lock();

                }
                sample_lock.unlock();
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
            printf("GNNmini::Engine[Dist.GPU.GCNimpl] running [%d] Epochs\n",
                   iterations);

        //      graph->print_info();
        // Note：该部分不涉及CUDA
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

            shuffle_vec(multi_train_nids[i]);
//        shuffle_vec(val_nids);
//        shuffle_vec(test_nids);


            //  nts::op::nts_local_shuffle(train_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);
            // nts::op::nts_local_shuffle(val_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);
            // nts::op::nts_local_shuffle(test_nids, graph->config->batch_size, graph->config->batch_size * pipeline_num);

        }

        InitCudaStream();

        int layer = graph->gnnctx->layer_size.size()-1;
//        if(graph->config->pushdown)
//            layer--;
        // TODO: 这里涉及了CUDA，以及多流，可以考虑每个GPU一个采样器
        for(int i = 0; i < num_devices; i++) {
            cudaSetUsingDevice(i);
            std::printf("train id size: %d\n", multi_train_nids[i].size());
            device_train_sampler[i] = new FastSampler(fully_rep_graph,multi_train_nids[i],layer,graph->config->batch_size,
                                                      graph->gnnctx->fanout, pipeline_num, device_cuda_stream[i]);
            std::printf("sampler train id size: %d\n", device_train_sampler[i]->work_range[1]);
        }
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
                    Forward(device_train_sampler[dev_id], 0, dev_id);
                }, device_id);
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
        auto wait_time = 0.0;
        for(int i = 0; i < pipeline_num; i++) {
            wait_time += wait_times[i];
        }
        printf("#run_time=%lf(s)\n", exec_time);
//        printf("all:%lf(s) prepro:%lf(s) pro:%lf(s) post:%lf(s) copy:%lf(s)\n",train_sampler->all_time,train_sampler->pre_pro_time, train_sampler->pro_time,train_sampler->post_pro_time,train_sampler->copy_gpu_time );
//        printf("test_time:%lf(s)\n",train_sampler->test_time);
        printf("#wait time: %lf(s)\n", wait_time);
        printf("#gather_feature_time=%lf(s)\n", gather_feature_time);
//        std::printf("cpu inclusiveTime: %lf\n", train_sampler->cs->cpu_inclusiveTime);
//        std::printf("inclusiveTime: %lf\n", train_sampler->cs->inclusiveTime);
//        std::printf("init layer time: %lf\n", train_sampler->init_layer_time);
//        std::printf("init co time: %lf\n", train_sampler->init_co_time);
        printf("#sample_time= %.4lf (s)\n", (sample_time));
        printf("#transfer_feature_time= %.4lf (s)\n", (transfer_feature_time));
        printf("#training_time= %.4lf (s)\n", training_time);
        delete active;
        printf("#average epoch time: %.4lf (s)\n", exec_time/iterations);
        printf("总传输节点数: %llu\n", Cuda_Stream::total_transfer_node);
//        printf("平均epoch传输节点数:%llu\n", Cuda_Stream::total_transfer_node/iterations);
        printf("%lu\n%lu\n", start_time, end_time);
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

};

#endif //GNNMINI_GCN_SAMPLE_ALL_MULTI_HPP
