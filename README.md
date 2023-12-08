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
./nts_run.sh ${cfg_file} 
```

ENGINE TYPE:
We list serveral example in the root dir for your reference
GCN:
gcn_reddit_sample.cfg
gcn_cora_sample.cfg
