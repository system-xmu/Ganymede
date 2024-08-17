#ifndef GEMINIFS_INTERNAL_H
#define GEMINIFS_INTERNAL_H
#include "geminifs_api.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}


__forceinline__ __device__ uint32_t
get_smid() {
     uint32_t ret;
     asm  ("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

__forceinline__ __device__ int
lane_id() {
    int lane_id = threadIdx.x & 0x1f;
    return lane_id;
}

__forceinline__ __device__
void warp_memcpy_4kB(void *dest_, const void *src_, uint32_t participating_mask) {
    uint64_t *dest = (uint64_t *)dest_;
    const uint64_t *src = (const uint64_t *)src_;
    int nr_participant = __popc(participating_mask);
    int lane = lane_id();
    int participant_id = __popc(participating_mask >> (32 - lane));
    for (size_t i = participant_id;
            i < 256;
            i += nr_participant * 16) {
        dest[i] = src[i];
        dest[i + nr_participant * 1] = src[i + nr_participant * 1];
        dest[i + nr_participant * 2] = src[i + nr_participant * 2];
        dest[i + nr_participant * 3] = src[i + nr_participant * 3];
        dest[i + nr_participant * 4] = src[i + nr_participant * 4];
        dest[i + nr_participant * 5] = src[i + nr_participant * 5];
        dest[i + nr_participant * 6] = src[i + nr_participant * 6];
        dest[i + nr_participant * 7] = src[i + nr_participant * 7];
        dest[i + nr_participant * 8] = src[i + nr_participant * 8];
        dest[i + nr_participant * 9] = src[i + nr_participant * 9];
        dest[i + nr_participant * 10] = src[i + nr_participant * 10];
        dest[i + nr_participant * 11] = src[i + nr_participant * 11];
        dest[i + nr_participant * 12] = src[i + nr_participant * 12];
        dest[i + nr_participant * 13] = src[i + nr_participant * 13];
        dest[i + nr_participant * 14] = src[i + nr_participant * 14];
        dest[i + nr_participant * 15] = src[i + nr_participant * 15];
    }
}

using CachePageId = size_t;
using FilePageId = size_t;
enum CachePage_State {
    CACHEPAGE_INVALID,
    CACHEPAGE_CLEAN,
    CACHEPAGE_DIRTY,
};

class CachePage {
public:
    __device__
    CachePage(void * const buf_, int never_evicted_): 
        buf(buf_),
        never_evicted(never_evicted_) { }

    __device__ virtual
    ~CachePage() { }

    __forceinline__ __device__ void
    write_back__no_lock(void *nvme_controller) {
        if (this->state == CACHEPAGE_DIRTY) {
            this->__write_back(nvme_controller, this->content_of);
            this->state = CACHEPAGE_CLEAN;
            __threadfence();
        }
    }
    __forceinline__ __device__ void
    read_in__no_lock(void *nvme_controller) {
        if ((this->content_of != this->assigned_to) || this->state == CACHEPAGE_INVALID) {
            this->write_back__no_lock(nvme_controller);

            // Here, the state is INVALID or CLEAN

            this->__read_in(nvme_controller, this->assigned_to);
            this->content_of = this->assigned_to;
            this->state = CACHEPAGE_CLEAN;
            __threadfence();
        }
    }

    __forceinline__ __device__ void
    sync(void *nvme_controller) {
        this->lock.acquire();
        this->write_back__no_lock(nvme_controller);
        this->lock.release();
    }

    __forceinline__ __device__ void
    set_dirty() {
        this->lock.acquire();
        this->state = CACHEPAGE_DIRTY;
        this->lock.release();
    }

    CachePageId cachepage_id;
    FilePageId content_of; // the filepage to which the data belongs
    FilePageId assigned_to; // the filepage specified by the pagecache mechanism
    enum CachePage_State state;
    cuda::binary_semaphore<cuda::thread_scope_device> lock;

    void * const buf;
    const int never_evicted;
private:
    virtual __device__ void __write_back(void *nvme_controller, FilePageId filepage_id) = 0;
    virtual __device__ void __read_in(void *nvme_controller, FilePageId filepage_id) = 0;
};

class PageCache {
public:
    __device__
    PageCache() { }

    __device__ virtual
    ~PageCache() { }

    virtual __device__ CachePageId
    acquire_page(FilePageId filepage_id, uint32_t participating_mask) = 0;

    virtual __device__ void
    set_page_dirty(CachePageId cachepage_id) = 0;

    virtual __device__ void
    release_page(FilePageId filepage_id, uint32_t participating_mask) = 0;

    virtual __device__ uint8_t *
    get_raw_page_buf(CachePageId cachepage_id) = 0;

    virtual __device__ int
    get_page_bit_num() = 0;
};


extern __host__ PageCache *
__internal__get_pagecache(
        uint64_t pagecache_capacity,
        int page_size,
        uint64_t size_of_virtual_space,
        CachePage *pages[]);

extern __host__ void
__internal__drop_pagecache(PageCache *);

template <typename T>
__global__ void
__run_device_lambda(T lambda) {
    lambda();
}

#define RUN_ON_DEVICE__MULTI_THREAD(CODE_BLOCK, nr_thread) { \
    __run_device_lambda<<<1, nr_thread>>>([=] __device__ () { \
        CODE_BLOCK \
    }); \
    cudaDeviceSynchronize(); \
}

#define RUN_ON_DEVICE(CODE_BLOCK) RUN_ON_DEVICE__MULTI_THREAD(CODE_BLOCK, 1)

#endif
