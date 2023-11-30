// #include <mpi.h>
// #include <iostream>
// #include <vector>

// int main(int argc, char* argv[]) {
//     MPI_Init(&argc, &argv);

//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     const int total_features = 6;  // 总特征数量
//     const int initial_feature_length = 12;  // 初始特征长度
//     const int split_feature_length = 4;  // 切分特征长度
//     const int features_per_node = total_features / size;  // 每个节点初始状态具有的特征数量

//     // 生成每个节点初始状态的特征
//     std::vector<float> local_features(features_per_node * initial_feature_length, 0.0);
//     for (int i = 0; i < features_per_node * initial_feature_length; ++i) {
//         local_features[i] = rank * features_per_node * initial_feature_length + i + 1;
//     }

//     if(rank == 1){
//         std::cout << "Rank " << rank << " has initial features: ";
//         for (int i = 0; i < features_per_node; ++i) {
//             std::cout << std::endl <<"feature " << i << " ";
//             for(int j = 0; j < initial_feature_length; ++j){
//                 std::cout << local_features[i * initial_feature_length + j] << " ";
//             }
//             std::cout << std::endl;
//         }
//         std::cout << std::endl;
//     }
//     // 本地切分特征数据
//     std::vector<float> split_features(features_per_node * split_feature_length * size, 0.0);
//     for (int i = 0; i < features_per_node; ++i) {
//         for (int j = 0; j < split_feature_length; ++j) {
//             split_features[i * size * split_feature_length + rank * split_feature_length + j] = 
//                 local_features[i * initial_feature_length + j];
//         }
//     }

//     // 输出本地切分后的特征
//     if(rank == 1){
//         std::cout << "Rank " << rank << " has split features: ";
//         for (int i = 0; i < features_per_node * split_feature_length * size; ++i) {
//             std::cout << split_features[i] << " ";
//         }
//         std::cout << std::endl;
//     }
//     // 使用MPI的AllReduce操作同步切分的特征数据
//     std::vector<float> global_split_features(features_per_node * split_feature_length * size, 0.0);
//     MPI_Allreduce(split_features.data(), global_split_features.data(), 
//                   features_per_node * split_feature_length * size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

//     // 输出全局同步后的切分特征
//     if (rank == 1) {
//         std::cout << "After AllReduce, global split features: ";
//         for (int i = 0; i < features_per_node * split_feature_length * size; ++i) {
//             std::cout << global_split_features[i] << " ";
//         }
//         std::cout << std::endl;
//     }

//     MPI_Finalize();
//     return 0;
// }


// #include <iostream>
// #include <vector>
// #include <mpi.h>

// int main() {
//     MPI_Init(NULL, NULL);

//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     const int total_features = 6;
//     const int feature_length = 12;
//     const int features_per_node = total_features / size;
//     const int split_feature_count = 3;
//     const int split_feature_length = feature_length / split_feature_count;

//     // 模拟每个节点拥有的本地特征
    
//     std::vector<std::vector<int>> local_features(features_per_node, std::vector<int>(feature_length, 0));
//     for (int i = 0; i < features_per_node; ++i) {
//         for (int j = 0; j < feature_length; ++j) {
//             local_features[i][j] = rank * features_per_node + i + 1;
//         }
//     }

//     // 输出每个节点的本地特征
//     if(rank == 0){
//         std::cout << "Process " << rank << ": Local Features:" << std::endl;
//         for (const auto& feature : local_features) {
//             for (int i : feature) {
//                 std::cout << i << " ";
//             }
//             std::cout << std::endl;
//         }
//     }
//     // 本地将每个特征拆分为三个切分特征
//     std::vector<int> split_features;
//     for (int i = 0; i < features_per_node; ++i) {
//         for (int j = 0; j < split_feature_count; ++j) {
//             for (int k = 0; k < split_feature_length; ++k) {
//                 split_features.push_back(local_features[i][j * split_feature_length + k]);
//             }
//         }
//     }

//     // 输出每个节点的切分特征
//     if(rank == 0){
//         std::cout << "Process " << rank << ": Split Features: ";
//         for (int i : split_features) {
//             std::cout << i << " ";
//         }
//         std::cout << std::endl;
//     }

//     // 每个节点进行第一次gather，使得每个节点拥有全部特征的一个切分特征
//     std::vector<int> gathered_split_features(feature_length * size, 0);
//     MPI_Gather(split_features.data(), feature_length / split_feature_count, MPI_INT,
//                gathered_split_features.data(), feature_length / split_feature_count, MPI_INT, 0, MPI_COMM_WORLD);

//     // 在根进程中进行同步
//     MPI_Barrier(MPI_COMM_WORLD);

//     // 输出每个节点收集到的切分特征
//     if (rank == 0) {
//         std::cout << "Root Process: Gathered Split Features: ";
//         for (int i : gathered_split_features) {
//             std::cout << i << " ";
//         }
//         std::cout << std::endl;
//     }

//     // // 在根进程中进行第二次gather，恢复初始状态
//     // if (rank == 0) {
//     //     std::vector<std::vector<int>> final_features(total_features, std::vector<int>(feature_length, 0));
//     //     MPI_Gather(gathered_split_features.data(), feature_length, MPI_INT,
//     //                final_features.data(), feature_length, MPI_INT, 0, MPI_COMM_WORLD);

//     //     // 输出最终结果
//     //     std::cout << "Root Process: Final Features:" << std::endl;
//     //     for (const auto& feature : final_features) {
//     //         for (int i : feature) {
//     //             std::cout << i << " ";
//     //         }
//     //         std::cout << std::endl;
//     //     }
//     // }

//     MPI_Finalize();
//     return 0;
// }

// #include <iostream>
// #include <vector>
// #include <mpi.h>

// int main() {
//     MPI_Init(NULL, NULL);

//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     const int block_size = 3;
//     const int total_blocks = 2;

//     // 创建 MPI_Datatype，描述非连续的数据布局
//     MPI_Datatype block_type;
//     MPI_Type_vector(total_blocks, block_size, block_size, MPI_INT, &block_type);
//     MPI_Type_commit(&block_type);

//     // 模拟每个进程的非连续数据
//     std::vector<int> send_data(total_blocks * block_size, 0);
//     for (int i = 0; i < total_blocks * block_size; ++i) {
//         send_data[i] = rank * total_blocks * block_size + i;
//     }

//     // 输出每个进程的发送数据
//     std::cout << "Process " << rank << ": Sending Data: ";
//     for (int i : send_data) {
//         std::cout << i << " ";
//     }
//     std::cout << std::endl;

//     // 接收缓冲区
//     std::vector<int> recv_data(size * total_blocks * block_size, 0);

//     // 进行 MPI_Gather 操作
//     MPI_Gather(send_data.data(), 1, block_type,
//                recv_data.data(), total_blocks * block_size, MPI_INT,
//                0, MPI_COMM_WORLD);

//     // 输出结果
//     if (rank == 0) {
//         std::cout << "Root Process: Gathered Data: ";
//         for (int i : recv_data) {
//             std::cout << i << " ";
//         }
//         std::cout << std::endl;
//     }

//     // 释放 MPI_Datatype
//     MPI_Type_free(&block_type);

//     MPI_Finalize();
//     return 0;
// }


#include <iostream>
#include <mpi.h>

int main() {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 10;
    const int k = 2;

    MPI_Datatype block_type;
    MPI_Type_vector(n / k, 1, k, MPI_INT, &block_type);
    MPI_Type_commit(&block_type);

    int send_data[n];
    for (int i = 0; i < n; ++i) {
        send_data[i] = rank * n + i;
    }

    // 创建接收缓冲区
    int recv_data[size * n / k];

    // 进行 MPI_Gather 操作
    MPI_Gather(send_data, 1, block_type,
               recv_data, n / k, MPI_INT,
               0, MPI_COMM_WORLD);

    // 输出结果
    if (rank == 0) {
        std::cout << "Root Process: Gathered Data: ";
        for (int i = 0; i < size * n / k; ++i) {
            std::cout << recv_data[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Type_free(&block_type);

    MPI_Finalize();
    return 0;
}





