/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   cuda_type.h
 * Author: wangqg
 *
 * Created on October 25, 2021, 9:39 AM
 */

#ifndef CUDA_TYPE_H
#define CUDA_TYPE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t VertexId_CUDA;
const int CUDA_NUM_THREADS=512;
const int CUDA_NUM_BLOCKS=32;

#ifdef __cplusplus
}
#endif

#endif /* CUDA_TYPE_H */
