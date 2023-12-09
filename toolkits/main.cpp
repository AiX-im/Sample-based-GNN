/*
Copyright (c) 2021-2022 Qiange Wang, Northeastern University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <signal.h>
#include <boost/stacktrace.hpp>
#include "GCN_CPU_SAMPLE.hpp"
#if CUDA_ENABLE
#include "GCN_SAMPLE_GPU.hpp"
#include "GCN_SAMPLE_ALLGPU.hpp"
#include "GCN_SAMPLE_PD_CACHE.hpp"
#include "GCN_SAMPLE_ALL_MULTI.hpp"
#include "GCN_SAMPLE_PC_MULTI.hpp"
#include "GS_SAMPLE_ALLGPU.hpp"
#include "GS_SAMPLE_PD_CACHE.hpp"
#include "GS_SAMPLE_PC_MULTI.hpp"
#include "GS_SAMPLE_CACHE.hpp"
#include "GAT_SAMPLE_ALL_GPU.hpp"
#include "GAT_SAMPLE_PD_CACHE.hpp"
#include "GAT_SAMPLE_ALL_MULTI.hpp"
#include "GAT_SAMPLE_PC_MULTI.hpp"
#endif

void segment_fault_handler(int signum) {
    std::fprintf(stderr, "Error: signal %d:\n", signum);
    std::cerr << boost::stacktrace::stacktrace();
    exit(signum);
}



int main(int argc, char **argv) {
    signal(SIGSEGV, segment_fault_handler);
  MPI_Instance mpi(&argc, &argv);
  if (argc < 2) {
    printf("configuration file missed \n");
    exit(-1);
  }

  double exec_time = 0;
  exec_time -= get_time();

  double run_time = 0;

  Graph<Empty> *graph;
  graph = new Graph<Empty>();
  graph->config->readFromCfgFile(argv[1]);
  if (graph->partition_id == 0)
    graph->config->print();

  //int iterations = graph->config->epochs;
  int iterations = graph->config->epochs;
  graph->replication_threshold = graph->config->repthreshold;

  if (graph->config->algorithm == std::string("GCNSAMPLESINGLE")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
        GCN_CPU_SAMPLE_impl *ntsGIN = new GCN_CPU_SAMPLE_impl(graph, iterations);
    ntsGIN->init_graph();
    ntsGIN->init_nn();
    run_time -= get_time();
    ntsGIN->run();
    run_time += get_time();
    if (graph->partition_id == 0) {
    printf("run_time=%lf(s)\n", run_time);
    }
  }

#if CUDA_ENABLE
  else if (graph->config->algorithm == std::string("GCNSAMPLEGPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_SAMPLE_GPU_impl *ntsGCN =
        new GCN_SAMPLE_GPU_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GCNSAMPLEALLGPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_SAMPLE_ALLGPU_impl *ntsGCN =
        new GCN_SAMPLE_ALLGPU_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GSSAMPLEALLGPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GS_SAMPLE_ALLGPU_impl *ntsGCN =
        new GS_SAMPLE_ALLGPU_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GCNSAMPLEPDCACHE")) {
      graph->load_directed(graph->config->edge_file, graph->config->vertices);
      graph->generate_backward_structure();
      GCN_SAMPLE_PD_CACHE_impl *ntsGCN =
              new GCN_SAMPLE_PD_CACHE_impl(graph, iterations);
      ntsGCN->init_graph();
      ntsGCN->init_nn();
      ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GCNSAMPLEALLMULTI")) {
      graph->load_directed(graph->config->edge_file, graph->config->vertices);
      graph->generate_backward_structure();
      auto *ntsGCN =
              new GCN_SAMPLE_ALL_MULTI_impl(graph, iterations);
      ntsGCN->init_graph();
      ntsGCN->init_nn();
      ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GCNSAMPLEPCMULTI")) {
      graph->load_directed(graph->config->edge_file, graph->config->vertices);
      graph->generate_backward_structure();
      auto *ntsGCN =
              new GCN_SAMPLE_PC_MULTI_impl(graph, iterations);
      ntsGCN->init_graph();
      ntsGCN->init_nn();
      ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GSSAMPLECACHE")) {
      graph->load_directed(graph->config->edge_file, graph->config->vertices);
      graph->generate_backward_structure();
      GS_SAMPLE_CACHE_impl *ntsGCN =
              new GS_SAMPLE_CACHE_impl(graph, iterations);
      ntsGCN->init_graph();
      ntsGCN->init_nn();
      ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GSSAMPLEPDCACHE")) {
      graph->load_directed(graph->config->edge_file, graph->config->vertices);
      graph->generate_backward_structure();
      GS_SAMPLE_PD_CACHE_impl *ntsGCN =
              new GS_SAMPLE_PD_CACHE_impl(graph, iterations);
      ntsGCN->init_graph();
      ntsGCN->init_nn();
      ntsGCN->run();  
  } else if (graph->config->algorithm == std::string("GSSAMPLEPCMULTI")) {
      graph->load_directed(graph->config->edge_file, graph->config->vertices);
      graph->generate_backward_structure();
      GS_SAMPLE_PC_MULTI_impl *ntsGCN =
              new GS_SAMPLE_PC_MULTI_impl(graph, iterations);
      ntsGCN->init_graph();
      ntsGCN->init_nn();
      ntsGCN->run();  
  } else if (graph->config->algorithm == std::string("GATSAMPLEPDCACHE")) {
      graph->load_directed(graph->config->edge_file, graph->config->vertices);
      graph->generate_backward_structure();
      auto *ntsGCN =
              new GAT_SAMPLE_PD_CACHE_impl(graph, iterations);
      ntsGCN->init_graph();
      ntsGCN->init_nn();
      ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GATSAMPLEALLGPU")) {
      graph->load_directed(graph->config->edge_file, graph->config->vertices);
      graph->generate_backward_structure();
      auto *ntsGCN =
              new GAT_SAMPLE_ALL_GPU_impl(graph, iterations);
      ntsGCN->init_graph();
      ntsGCN->init_nn();
      ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GATSAMPLEALLMULTI")) {
      graph->load_directed(graph->config->edge_file, graph->config->vertices);
      graph->generate_backward_structure();
      auto *ntsGCN =
              new GAT_SAMPLE_ALL_MULTI_impl(graph, iterations);
      ntsGCN->init_graph();
      ntsGCN->init_nn();
      ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GATSAMPLEPCMULTI")) {
      graph->load_directed(graph->config->edge_file, graph->config->vertices);
      graph->generate_backward_structure();
      auto *ntsGCN =
              new GAT_SAMPLE_PC_MULTI_impl(graph, iterations);
      ntsGCN->init_graph();
      ntsGCN->init_nn();
      ntsGCN->run();
  } 
#endif
  exec_time += get_time();
  if (graph->partition_id == 0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  delete graph;

  //    ResetDevice();

  return 0;
}
