/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Vector addition: C = A + B.
 *
 * This sample replaces the device allocation in the vectorAddDrvsample with
 * cuMemMap-ed allocations.  This sample demonstrates that the cuMemMap api
 * allows the user to specify the physical properties of their memory while
 * retaining the contiguos nature of their access, thus not requiring a change
 * in their program structure.
 *
 */

// Includes
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>
#include <cuda.h>
#include <cufile.h>
#include "cufile_sample_utils.h"
#include <iomanip> 
// includes, project
//#include "helper_cuda_drvapi.h"
//#include "helper_functions.h"

// includes, CUDA
#include <builtin_types.h>
#include <cuda.h>
#include <vector>
#include <thread>
#include <chrono>  



// CUDA Driver API errors
static const char *_cudaGetErrorEnum(CUresult error) {
  static char unknown[] = "<unknown>";
  const char *ret = NULL;
  cuGetErrorName(error, &ret);
  return ret ? ret : unknown;
}


template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

static size_t round_up(size_t x, size_t y)
{
    return ((x + y - 1) / y) * y;
}



using namespace std;

typedef struct cfg {
        int gpu;
        CUfileHandle_t cf_handle;
        size_t offset;
        int nthreads;
} cfg_t;
#define MAX_N_THREADS 700
pthread_t threads[MAX_N_THREADS];
const char *TESTFILE;

#ifdef __cplusplus
extern "C" {
extern void vectorAdd(const float *A, const float *B, float *C,
                          int numElements);
}
#endif
#define MAX_READ_IO_NUM (2000000)

u_int64_t file_size = 1LL << 33; // 8GB
size_t pageSize = 4096; // 1<<12

static void* thread_cuRead(void* arg)
{
       
        CUfileError_t status;
        int ret;

        cfg_t *cfg = (cfg_t *)arg;
        int cnt = 0;
        for (u_int64_t offset = cfg->offset; offset < file_size; offset += cfg->nthreads*pageSize)
        {
                if(cnt == MAX_READ_IO_NUM)
			break;
                cnt++;
                void* d_buffer = NULL;
                // Malloc
                check_cudaruntimecall(cudaSetDevice(cfg->gpu));
                check_cudaruntimecall(cudaMalloc(&d_buffer, pageSize));

                auto start = std::chrono::high_resolution_clock::now(); 
                
                // Register buffer
                status = cuFileBufRegister((void*)d_buffer, pageSize, 0);
                if (status.err != CU_FILE_SUCCESS) {
                        ret = -1;
                        std::cerr << "Register buffer failed:"
                                << cuFileGetErrorString(status) << std::endl;
                        exit(EXIT_FAILURE);
                }

                // Read pageSize I/O
                ret = cuFileRead(cfg->cf_handle, (void*)d_buffer, pageSize, offset, 0);
                if (ret < 0) 
                {
                        if (IS_CUFILE_ERR(ret))
                                std::cerr << "read failed : "<< cuFileGetErrorString(ret) << std::endl;
                        else
                        {
                                std::cerr << "read failed : "<< cuFileGetErrorString(errno) << std::endl;
                                exit(1);
                        }
                                
                } else 
                {
                        // std::cout << "read bytes to d_buffer:" << ret << std::endl;
                        ret = 0;
                }

                // Deregister buffer 
                status =cuFileBufDeregister(d_buffer);
                if (status.err != CU_FILE_SUCCESS) 
                {
                        ret = -1;
                        std::cerr << "Deregister buffer failed:"<< cuFileGetErrorString(status) << std::endl;
                        exit(EXIT_FAILURE);
                }
                
                auto end = std::chrono::high_resolution_clock::now();  
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double rtime = static_cast<double>(elapsed.count());
                printf("Thread read time: %.3f us\n",rtime);
                
                if(d_buffer)
                        check_cudaruntimecall(cudaFree(d_buffer));

        }
        
       

        return NULL;
}

// Host code
int main(int argc, char **argv)
{
        int fd=-1, ret;
        CUfileError_t status;
        CUfileDescr_t cf_descr;
        CUfileHandle_t cf_handle;
      


        if(argc < 4) {
                std::cerr << argv[0] << " gpuId nthreads <filepath> "<< std::endl;
                exit(EXIT_FAILURE);
        }
        int gpuId = atoi(argv[1]);
        int nthreads = atoi(argv[2]);
        TESTFILE = argv[3];

        status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "cufile driver open error: "
                        << cuFileGetErrorString(status) << std::endl;
                exit(EXIT_FAILURE);
        }

        // std::cout << "opening file " << TESTFILE << std::endl;

        // opens file 
        ret = open(TESTFILE, O_CREAT | O_RDWR | O_DIRECT, 0664);
        if (ret < 0) {
                std::cerr << "file open error:"
                        << cuFileGetErrorString(errno) << std::endl;
                exit(EXIT_FAILURE);
        }
        fd = ret;

       


        memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
        cf_descr.handle.fd = fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "file register error:"
                        << cuFileGetErrorString(status) << std::endl;
                close(fd);
                exit(EXIT_FAILURE);
        }

        size_t offset = 0;
        for (int i = 0; i < nthreads; i++)
        {
                cfg_t cfg;
                pthread_t thread;
                cfg.gpu = gpuId;
                cfg.cf_handle = cf_handle;
                cfg.offset = offset + pageSize * i;     // init thread offset
                cfg.nthreads = nthreads;
                pthread_create(&thread, NULL, &thread_cuRead, &cfg);
                threads[i] = thread;
        }
        for (int i = 0; i < nthreads; i++)
        {
                pthread_join(threads[i], NULL);
        }
        


        // deregister the device memory
        status = cuFileDriverClose();
        if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "cuFileDriverClose failed:"
                        << cuFileGetErrorString(status) << std::endl;
        }
        close(fd);

        return 0;


}
