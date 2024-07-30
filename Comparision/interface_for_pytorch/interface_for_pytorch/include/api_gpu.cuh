#ifndef API_GPU_CU_H
#define API_GPU_CU_H

#include <errno.h>
#include <string.h>
#include "fs_calls.cuh"
#include "host_loop.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(err)  __checkCudaErrors (err, __FILE__, __LINE__)
#endif

typedef unsigned char uchar;  

#endif // API_GPU_CU_H