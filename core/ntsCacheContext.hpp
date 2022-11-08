//
// Created by aixin on 2022/7/10.
//

#ifndef NTSCACHE_HPP
#define NTSCACHE_HPP

#include <vector>
#include <cstdio>
#include <mpi.h>
#include <string>
#include <cstring>
#include <algorithm>
#include "FullyRepGraph.hpp"

class ntsCache
{
public:
    VertexId* sample_num;
    std::vector<VertexId> sample_sort;
    std::vector<std::vector<VertexId>> partition_cache;
    std::vector<VertexId> cacheflag;
    std::vector<std::vector<VertexId>> normalnode;
    std::map<VertexId, int> map_cache;
    VertexId cacheBoundary;
    VertexId fanoutBoundary;   
    float cacheRatio;
    float fanoutRatio;
    int cacheSamplenum;
    int cachenum;
    ntsCache(float _cacheRatio, float _fanoutRatio, int vertices, int partitions){
        cacheRatio = _cacheRatio;
        fanoutRatio = _fanoutRatio;
        cacheBoundary = cacheRatio * vertices;
        sample_sort.resize(vertices);
        cacheflag.resize(vertices);
        partition_cache.resize(partitions);
        normalnode.resize(partitions);
        cachenum = 0;
    }
    ~ntsCache() {
    }
};


#endif //NTSCACHE_HPP
