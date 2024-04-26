**NeutronOrch** is a system for sample-based GNN training that incorporates a layer-based task orches-trating method and ensures balanced utilization of the CPU and GPU. NeutronOrch distinguishes itself from other GNN training frameworks with the following new properties:

 * NeutronOrch use a hotness-aware layer-based task orchestrating method that effectively leverages the computation and memory resources of the GPU-CPU heterogeneous system.
 * NeutronOrch integrates the pytorch automatic differentiation library libtorch and tensorflow to support automatic differentiation (automatic backpropagation) across workers. 
 * NeutronOrch use a super-batch pipelined training, which fully overlaps different tasks on heterogeneous resources while strictly guaranteeing bounded staleness.


Currently NeutronOrch is under refactoring. We will release all features of NeutronOrch soon.



## Quick Start

A compiler supporting **OpenMP** and **C++11** features (e.g. lambda expressions, multi-threading, etc.) is required.

**cmake** >=3.14.3

**MPI** for inter-process communication 

**cuda** > 11.0 for GPU based graph operation.

**TBB** for C++ 17 feature.

**libnuma** for NUMA-aware memory allocation.


```
sudo apt install libnuma-dev
```

**libtorch** version > 1.11 with gpu support for nn computation

unzip the **libtorch** package in the root dir of **NeutronOrch** and change CMAKE_PREFIX_PATH in "CMakeList.txt"to your own path


configure PATH and LD_LIBRARY_PATH for **cuda** and **mpi**
```
export CUDA_HOME=/usr/local/cuda
export MPI_HOME=/path/to/your/mpi
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$MPI_HOME/bin:$CUDA_HOME/bin:$PATH
```

**clang-format** is optional for auto-formatting: 
```shell
sudo apt install clang-format
```

To build:
```shell
mkdir build

cd build

cmake ..

make -j4
```


### To run NeutronOrch:
```
./nto_run.sh ${cfg_file} 
```

ENGINE TYPE:
We list serveral example in the root dir for your reference
GCN:
gcn_reddit_sample.cfg
gcn_cora_sample.cfg


### Configurations in cfg file:
NeutronOrch needs following configurations to run:
- `ALGORITHM`: "xx_ALL_GPU" utilizes GPU exclusively for the sampling-gathering-training process, "xx_PDCACHE" offloads hot vertices computations to the CPU, and "xx_MULIT" means multi GPU version.
- `VERTICES`: the number of total vertices.
- `LAYERS`: the model depth and the dimensions of different layers.
- `FANOUT`: the number of sampled neighbor.
- `BATCH_SIZE`: the number of vertices in every batch.
- `EPOCHS`: epoch number.
- `EDGE_FILE`: an edge list file, used to store the graph structure.
- `FEATURE_FILE`: a file containing the feature of each node, the first number in each line indicates the node number, followed by the feature of the node.
- `LABEL_FILE`: a file containing the label of each node, the first number in each line indicates the node number, followed by the classification number of the node.
- `MASK_FILE`: a file containing the mask of each node, the first number in each line indicates the node number, followed by the mask of node (train, val, test).
- `model parameter` including `LEARN_RATE` ,`DECAY_EPOCH` ,`WEIGHT_DECAY` ,`DECAY_RATE` , and `DROP_RATE`.
- `PIPELINE_NUM`: represents the number of batches contained in each super-batch in super-batch pipelining.
- `CACHE_RATE`: represents the proportion of hot vertices computed by the CPU to all vertices in the initial state.
- `CACHE`: indicates whether GPU memory caching of vertex features is used
- `FEATURE_CACHE_RATE`: represents the number of vertex features cached by the GPU in the initial state.

We provide a python script to convert some commonly used data sets, please refer to [`data/generate_nts_dataset.py`](https://github.com/iDC-NEU/NeutronStarLite/blob/master/data/generate_nts_dataset.py) for details.


