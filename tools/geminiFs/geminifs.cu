//#include <stdio.h>
//#include <assert.h>
//#include <sys/types.h>
//#include <unistd.h>

#include <cuda/atomic>
#include <cuda/semaphore>
#include <cuco/static_map.cuh>
#include <cuco/static_set.cuh>

#include <cooperative_groups.h>

#include "geminifs.h"

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
    uint32_t lane = lane_id();
    uint32_t participant_id = __popc(participating_mask >> (32 - lane));

    for (size_t i = participant_id;
            i < 32 / nr_participant;
            i += 16 * nr_participant) {
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

struct MyLinklistNode {
    MyLinklistNode *pre, *suc;
};
template <typename T>
struct MyLinklistNodeD: public MyLinklistNode {
    T v;
};
class MyLinklist {
public:
    __device__ MyLinklist() {
        this->head.suc = &(this->tail);
        this->tail.pre = &(this->head);
    }
    __inline__ __device__ void
    push(MyLinklistNode *new_node) {
        this->insert_after(&(this->head), new_node);
    }
    __inline__ __device__ void
    enqueue(MyLinklistNode *new_node) {
        this->insert_after(this->tail.pre, new_node);
    }
    __inline__ __device__ MyLinklistNode *
    pop() {
        auto *ret_node = this->head.suc;
        if (ret_node != &(this->tail)) {
            this->detach_node(ret_node);
            return ret_node;
        }
        assert(0);
        return ret_node;
    }
    __inline__ __device__ void
    detach_node(MyLinklistNode *n) {
        (n->pre)->suc = n->suc;
        (n->suc)->pre = n->pre;
    }
private:
    __forceinline__ __device__ void
    insert_after(MyLinklistNode *insert_point, MyLinklistNode *n) {
        n->pre = insert_point;
        n->suc = insert_point->suc;
        insert_point->suc->pre = n;
        insert_point->suc = n;
    }
    MyLinklistNode head, tail;
};

using CachePageId = size_t;
using FilePageId = size_t;

enum CachePage_State {
    CACHEPAGE_INVALID,
    CACHEPAGE_CLEAN,
    CACHEPAGE_DIRTY,
};
class CachePage {
public:
    __forceinline__ __device__ void
    write_back__no_lock(void *nvme_controller) {
        if (this->state == CACHEPAGE_DIRTY) {
            this->nvme_write(nvme_controller, this->get_nvmeofst(this->content_of));
            this->state = CACHEPAGE_CLEAN;
            __threadfence();
        }
    }
    __forceinline__ __device__ void
    read_in__no_lock(void *nvme_controller) {
        if ((this->content_of != this->assigned_to) || this->state == CACHEPAGE_INVALID) {
            this->write_back__no_lock(nvme_controller);

            // Here, the state is INVALID or CLEAN

            this->nvme_read(nvme_controller, this->get_nvmeofst(this->assigned_to));
            this->content_of = this->assigned_to;
            this->state = CACHEPAGE_CLEAN;
            __threadfence();
        }
    }

    virtual __device__ nvme_ofst_t get_nvmeofst(FilePageId filepage_id) = 0;
    virtual __device__ void nvme_write(void *nvme_controller, nvme_ofst_t nvme_ofst) = 0;
    virtual __device__ void nvme_read(void *nvme_controller, nvme_ofst_t nvme_ofst) = 0;

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
    void *buf;
    enum CachePage_State state;
    cuda::binary_semaphore<cuda::thread_scope_device> lock;
};

class CachePage_Allocated: public CachePage {
public:
    __device__ CachePage_Allocated(int page_size) {
        cdpErrchk(cudaMalloc(&(this->buf), page_size));
    }
    __device__ ~CachePage_Allocated() {
        cdpErrchk(cudaFree(this->buf));
    }
    __device__ nvme_ofst_t get_nvmeofst(FilePageId filepage_id) {return -1;}
    __device__ void nvme_write(void *nvme_controller, nvme_ofst_t nvme_ofst) { }
    __device__ void nvme_read(void *nvme_controller, nvme_ofst_t nvme_ofst) {assert(0);}
};

class PageCache {
public:
    virtual __device__ CachePageId
    acquire_page(FilePageId filepage_id) = 0;

    virtual __device__ void
    set_page_dirty(CachePageId cachepage_id) = 0;

    virtual __device__ void
    release_page(FilePageId filepage_id) = 0;

    virtual __device__ uint8_t *
    get_raw_page_buf(CachePageId cachepage_id) = 0;

    virtual __device__ int
    get_page_bit_num() = 0;
};

class PageCacheImpl__info1 {
public:
    FilePageId filepage_id;
    cuda::counting_semaphore<cuda::thread_scope_device> wait_for_evicting;
    int nr_waiting;
};

template <typename MapRef1, typename MapRef2, typename MapRef3>
class PageCacheImpl: public PageCache {
public:
    __device__ PageCacheImpl(uint64_t pagecache_capacity,
            int page_size,
            bool no_backing_file,
            uint64_t size_of_virtual_space,
            MapRef1 map_ref1,
            MapRef2 map_ref2,
            MapRef3 map_ref3):
        filepages__waiting_for_evicting(map_ref1),
        nr_waiting(0),
        filepage__to__cachepage(map_ref2),
        zero_reffed_filepages(map_ref3) {
            if (no_backing_file)
                assert(size_of_virtual_space == pagecache_capacity);

            uint64_t nr_page = pagecache_capacity / page_size;

            this->page_size = page_size;
            this->nr_page = nr_page;

            this->pages = new CachePage * [nr_page];
            this->pages_ref = new uint64_t[nr_page];


            if (no_backing_file) {
                for (CachePageId cachepage_id = 0; cachepage_id < nr_page; cachepage_id++) {
                    FilePageId filepage_id = cachepage_id;

                    this->pages[cachepage_id] = new CachePage_Allocated(page_size);
                    this->pages[cachepage_id]->cachepage_id = cachepage_id;
                    this->pages[cachepage_id]->content_of = filepage_id;
                    this->pages[cachepage_id]->assigned_to = filepage_id;
                    this->pages_ref[cachepage_id] = 1; // the Ref Count is at lease 1, thus the page won't be evicted.
                    this->__insert__filepage__mapping_to__cachepage(filepage_id, cachepage_id);
                    this->pages[cachepage_id]->state = CACHEPAGE_CLEAN;

                    printf("%llx\n", filepage_id);
                }
            } else {
                for (CachePageId cachepage_id = 0; cachepage_id < nr_page; cachepage_id++) {
                    FilePageId filepage_id = cachepage_id;

                    this->pages[cachepage_id] = new CachePage_Allocated(page_size);
                    this->pages[cachepage_id]->cachepage_id = cachepage_id;
                    this->pages[cachepage_id]->content_of = filepage_id;
                    this->pages[cachepage_id]->assigned_to = filepage_id;
                    this->__insert__filepage__mapping_to__cachepage(filepage_id, cachepage_id);
                    this->pages_ref[cachepage_id] = 0;
                    this->__insert__zero_reffed_filepage(filepage_id);

                    this->pages[cachepage_id]->state = CACHEPAGE_INVALID;

                    printf("%llx\n", filepage_id);
                }
            }
        }

    __device__ ~PageCacheImpl() {
        assert(0);
        for (CachePageId cachepage_id = 0;
                cachepage_id < nr_page;
                cachepage_id++)
            delete this->pages[cachepage_id];

        delete this->pages_ref;
        delete this->pages;
    }

    __forceinline__ __device__ CachePageId
    acquire_page(FilePageId filepage_id) {
        CachePageId cachepage_id;

        uint32_t mask = __activemask();
        mask = __match_any_sync(mask, (uint64_t)this);
        mask = __match_any_sync(mask, filepage_id);
        uint32_t warp_leader = __ffs(mask) - 1;
        int lane = lane_id();
        if (lane == warp_leader)
            cachepage_id = this->__acquire_page_for_warp_leader(filepage_id);
        cachepage_id = __shfl_sync(mask, cachepage_id, warp_leader);
        return cachepage_id;
    }

    __forceinline__ __device__ void
    set_page_dirty(CachePageId cachepage_id) {
        uint32_t mask = __activemask();
        mask = __match_any_sync(mask, (uint64_t)this);
        mask = __match_any_sync(mask, cachepage_id);
        uint32_t warp_leader = __ffs(mask) - 1;
        int lane = lane_id();
        if (lane == warp_leader)
            this->pages[cachepage_id]->set_dirty();
    }

    __forceinline__ __device__ void
    release_page(FilePageId filepage_id) {
        uint32_t mask = __activemask();
        mask = __match_any_sync(mask, (uint64_t)this);
        mask = __match_any_sync(mask, filepage_id);
        uint32_t warp_leader = __ffs(mask) - 1;
        int lane = lane_id();
        if (lane == warp_leader)
            this->__release_page_for_warp_leader(filepage_id);
    }

    __forceinline__ __device__ uint8_t *
    get_raw_page_buf(CachePageId cachepage_id) {
        return (uint8_t *)this->pages[cachepage_id]->buf;
    }

    __forceinline__ __device__ int
    get_page_bit_num() {
        return __popc(this->page_size - 1);
    }
private:
    __forceinline__ __device__ CachePageId
    __acquire_page_for_warp_leader(FilePageId filepage_id) {
        CachePageId ret;

        this->pagecache_lock.acquire();

        ret = this->__get__cachepage_id(filepage_id);
        if (ret != -1) {
            // Page Hit!
            size_t cur_ref = (++(this->pages_ref[ret]));
            if (cur_ref == 1)
                this->__erase__zero_reffed_filepage(filepage_id);

            __threadfence();
            this->pagecache_lock.release();

            this->pages[ret]->lock.acquire();
            this->pages[ret]->read_in__no_lock(this->nvme_controller);
            this->pages[ret]->lock.release();

            return ret;
        }

        // Miss
        // ret == -1

        FilePageId evicted_filepage_id = this->__pop__zero_reffed_filepage_id();
        if (evicted_filepage_id != -1) {
            // Here we find a zero-reffed page to be evicted
            ret = this->__get__cachepage_id(evicted_filepage_id);
            ++(this->pages_ref[ret]);

            this->__erase__filepage__mapping(evicted_filepage_id);
            this->__insert__filepage__mapping_to__cachepage(filepage_id, ret);

            this->pages[ret]->assigned_to = filepage_id;

            __threadfence();
            this->pagecache_lock.release();

            this->pages[ret]->lock.acquire();
            this->pages[ret]->read_in__no_lock(this->nvme_controller);
            this->pages[ret]->lock.release();
            return ret;
        }

        // There is no page to be evicted now.
        // Waitting for a zero-reffed one.

        PageCacheImpl__info1 *leaders_waiting_for_evicting =
            this->__is_filepage_waiting_for_evicting(filepage_id);
        bool has_quota_to_wait = this->__has_quota_to_wait_for_evicting();

        if (leaders_waiting_for_evicting == nullptr && !has_quota_to_wait) {
            this->pagecache_lock.release();
            __nanosleep(200);
            return this->__acquire_page_for_warp_leader(filepage_id);
        }

        if (leaders_waiting_for_evicting == nullptr && has_quota_to_wait) {
            leaders_waiting_for_evicting = new PageCacheImpl__info1();
            leaders_waiting_for_evicting->filepage_id = filepage_id;
            leaders_waiting_for_evicting->nr_waiting = 0;
            this->__insert__filepage_waiting_for_evicting(filepage_id,
                    leaders_waiting_for_evicting);
        }
        leaders_waiting_for_evicting->nr_waiting++;
        __threadfence();
        this->pagecache_lock.release();

        // Sleep-------------------------------------------------------------------
        leaders_waiting_for_evicting->wait_for_evicting.acquire();
        // Be awaken because the filepage is assigned-----------------------------

        this->pagecache_lock.acquire();
        ret = this->__get__cachepage_id(filepage_id);
        assert(ret != -1);

        auto cur_waiting = leaders_waiting_for_evicting->nr_waiting;
        if (cur_waiting == 1) {
            delete leaders_waiting_for_evicting;
        } else {
            leaders_waiting_for_evicting->nr_waiting--;
        }

        __threadfence();
        this->pagecache_lock.release();
        this->pages[ret]->lock.acquire();
        this->pages[ret]->read_in__no_lock(this->nvme_controller);
        this->pages[ret]->lock.release();
        return ret;
    }

    __forceinline__ __device__ void
    __release_page_for_warp_leader(FilePageId filepage_id) {
        this->pagecache_lock.acquire();
        CachePageId cachepage_id = this->__get__cachepage_id(filepage_id);
        if ((--(this->pages_ref[cachepage_id])) == 0) {
            // The last one reffing the cachepage exits.
            PageCacheImpl__info1 *p = this->__pop__filepage_waiting_for_evicting();
            if (p) {
                // There is someone waiting for evicting
                this->pages[cachepage_id]->assigned_to = p->filepage_id;
                this->pages_ref[cachepage_id] += p->nr_waiting;
                this->__erase__filepage__mapping(filepage_id);
                this->__insert__filepage__mapping_to__cachepage(p->filepage_id, cachepage_id);

                auto nr_waiting = p->nr_waiting;

                __threadfence();
                this->pagecache_lock.release();

                this->pages[cachepage_id]->lock.acquire();
                this->pages[cachepage_id]->write_back__no_lock(this->nvme_controller);
                this->pages[cachepage_id]->lock.release();

                for (size_t i = 0; i < nr_waiting; i++)
                    p->wait_for_evicting.release();

                return;
            }

            // Nobody waits for evicting
            this->__insert__zero_reffed_filepage(filepage_id);
        }
        __threadfence();
        this->pagecache_lock.release();
    }
//----------------------------------------------------------
    __forceinline__ __device__ PageCacheImpl__info1 *
    __is_filepage_waiting_for_evicting(FilePageId filepage_id) {
        constexpr auto cg_size = this->filepages__waiting_for_evicting.cg_size;
        auto tile = cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());
        auto found = this->filepages__waiting_for_evicting.find(tile, filepage_id);
        if (found != this->filepages__waiting_for_evicting.end()) {
            auto *n = static_cast<MyLinklistNodeD<PageCacheImpl__info1 *> *>(found->second);
            return n->v;
        } else
            return nullptr;
    }

    __forceinline__ __device__ bool
    __has_quota_to_wait_for_evicting() {
        return this->nr_waiting < this->filepages__waiting_for_evicting.capacity();
    }

    __forceinline__ __device__ void
    __insert__filepage_waiting_for_evicting(FilePageId filepage_id,
                                             PageCacheImpl__info1 *p) {
        auto *n = new MyLinklistNodeD<PageCacheImpl__info1 *>();
        n->v = p;
        this->filepages__waiting_for_evicting__list.enqueue(n);
        this->nr_waiting++;

        constexpr auto cg_size = this->filepages__waiting_for_evicting.cg_size;
        auto tile = cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());
        assert(this->filepages__waiting_for_evicting.insert(tile, cuco::pair{filepage_id, n}));
    }

    __forceinline__ __device__ PageCacheImpl__info1 *
    __pop__filepage_waiting_for_evicting() {
        // get and erase
        auto *n = static_cast<MyLinklistNodeD<PageCacheImpl__info1 *> *>(this->filepages__waiting_for_evicting__list.pop());
        auto ret = n->v;
        this->nr_waiting--;

        constexpr auto cg_size = this->filepages__waiting_for_evicting.cg_size;
        auto tile = cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());
        assert(this->filepages__waiting_for_evicting.erase(tile, ret->filepage_id));

        return ret;
    }
//------------------------------------------------------------
    __forceinline__ __device__ void
    __insert__filepage__mapping_to__cachepage(FilePageId filepage_id,
                                              CachePageId cachepage_id) {
        constexpr auto cg_size = this->filepage__to__cachepage.cg_size;
        auto tile = cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());
        assert(this->filepage__to__cachepage.insert(tile, cuco::pair{filepage_id, cachepage_id}));
    }

    __forceinline__ __device__ void
    __erase__filepage__mapping(FilePageId filepage_id) {
        constexpr auto cg_size = this->filepage__to__cachepage.cg_size;
        auto tile = cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());
        assert(this->filepage__to__cachepage.erase(tile, filepage_id));
    }

    __forceinline__ __device__ CachePageId
    __get__cachepage_id(FilePageId filepage_id) {
        constexpr auto cg_size = this->filepage__to__cachepage.cg_size;
        auto tile = cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());
        auto found = this->filepage__to__cachepage.find(tile, filepage_id);
        if (found != this->filepage__to__cachepage.end())
            return found->second;
        else
            return -1;
    }
//-----------------------------------------------------------
    __forceinline__ __device__ void
    __insert__zero_reffed_filepage(FilePageId filepage_id) {
        auto *n = new MyLinklistNodeD<FilePageId>();
        n->v = filepage_id;
        this->zero_reffed_filepages__list.enqueue(n);

        constexpr auto cg_size = this->zero_reffed_filepages.cg_size;
        auto tile = cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());

        assert(this->zero_reffed_filepages.insert(tile, cuco::pair{filepage_id, n}));
    }

    __forceinline__ __device__ void
    __erase__zero_reffed_filepage(FilePageId filepage_id) {
        constexpr auto cg_size = this->zero_reffed_filepages.cg_size;
        auto tile = cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());

        MyLinklistNodeD<FilePageId> *n = nullptr;
        auto found = this->zero_reffed_filepages.find(tile, filepage_id);
        if (found != this->zero_reffed_filepages.end())
            n = static_cast<MyLinklistNodeD<FilePageId> *>(found->second);

        assert(n);

        assert(this->zero_reffed_filepages.erase(tile, filepage_id));
        this->zero_reffed_filepages__list.detach_node(n);
        delete n;
    }

    __forceinline__ __device__ FilePageId
    __pop__zero_reffed_filepage_id() {
        // get and erase
        auto *n = static_cast<MyLinklistNodeD<FilePageId> *>(this->zero_reffed_filepages__list.pop());
        auto filepage_id = n->v;
        delete n;

        constexpr auto cg_size = this->zero_reffed_filepages.cg_size;
        auto tile = cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());
        assert(this->zero_reffed_filepages.erase(tile, filepage_id));

        return filepage_id;
    }
//-----------------------------------------------------------

    cuda::binary_semaphore<cuda::thread_scope_device> pagecache_lock;

    void *nvme_controller;

    int page_size;

    uint64_t nr_page;

    CachePage **pages;
    uint64_t *pages_ref;

    size_t nr_waiting;
    MapRef1 filepages__waiting_for_evicting;
    MyLinklist filepages__waiting_for_evicting__list;

    MapRef2 filepage__to__cachepage;

    MapRef3 zero_reffed_filepages;
    MyLinklist zero_reffed_filepages__list;
};

template <typename MapRef1, typename MapRef2, typename MapRef3>
static __global__ void
host_open_geminifs_file_for_device_2(PageCache **pagecache_dev,
        uint64_t pagecache_capacity,
        int page_size,
        bool no_backing_file,
        uint64_t size_of_virtual_space,
        MapRef1 map_ref1,
        MapRef2 map_ref2,
        MapRef3 map_ref3) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0)
        return;

    auto *pagecache = new PageCacheImpl<MapRef1, MapRef2, MapRef3>(pagecache_capacity,
            page_size,
            no_backing_file,
            size_of_virtual_space,
            map_ref1,
            map_ref2,
            map_ref3);
    *pagecache_dev = pagecache;
}

static PageCache *
host_open_geminifs_file_for_device_1(uint64_t pagecache_capacity,
        int page_size,
        bool no_backing_file,
        uint64_t size_of_virtual_space) {
    PageCache **pagecache_ptr_dev;
    gpuErrchk(cudaMalloc(&pagecache_ptr_dev, sizeof(&pagecache_ptr_dev)));

    uint64_t nr_page = pagecache_capacity / page_size;

    // Initial host part of containers
    FilePageId constexpr empty_FilePageId_sentinel = -1;
    CachePageId constexpr empty_CachePageId_sentinel = -1;
    MyLinklistNode constexpr *sentinel = nullptr;

    auto *filepages__waiting_for_evicting = new cuco::static_map{nr_page,
        cuco::empty_key{empty_FilePageId_sentinel},
        cuco::empty_value{sentinel}};
    auto filepages__waiting_for_evicting__ref =
        filepages__waiting_for_evicting->ref(cuco::insert, cuco::find, cuco::erase);

    auto *filepage__to__cachepage = new cuco::static_map{nr_page,
        cuco::empty_key{empty_FilePageId_sentinel},
        cuco::empty_value{empty_CachePageId_sentinel}};
    auto filepage__to__cachepage__ref =
        filepage__to__cachepage->ref(cuco::insert, cuco::find, cuco::erase);

    auto *zero_reffed_filepages = new cuco::static_map{nr_page,
        cuco::empty_key{empty_FilePageId_sentinel},
        cuco::empty_value{sentinel}};
    auto zero_reffed_filepages__ref =
        zero_reffed_filepages->ref(cuco::insert, cuco::find, cuco::erase);


    //host_open_geminifs_file_for_device_2<<<1, 1>>>(pagecache_ptr_dev,
    //        pagecache_capacity,
    //        page_size,
    //        no_backing_file,
    //        size_of_virtual_space,
    //        filepages__waiting_for_evicting__ref,
    //        filepage__to__cachepage__ref,
    //        zero_reffed_filepages__ref);

    PageCache *pagecache_dev;
    gpuErrchk(cudaMemcpy(&pagecache_dev, pagecache_ptr_dev, sizeof(pagecache_dev),
                cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(pagecache_ptr_dev));
    return pagecache_dev;
}

dev_fd_t
host_open_geminifs_file_for_device(host_fd_t host_fd, uint64_t pagecache_capacity) {
    struct geminiFS_hdr *hdr = host_fd;

    nvme_ofst_t *l1__dev;
    gpuErrchk(cudaMalloc(&l1__dev, hdr->first_block_base));

    struct geminiFS_hdr *hdr__file = (struct geminiFS_hdr *)malloc(hdr->first_block_base);

    assert((off_t)(-1) !=
            lseek(hdr->fd, 0, SEEK_SET)
          );
    assert(hdr->first_block_base ==
            read(hdr->fd, hdr__file, hdr->first_block_base)
          );
    gpuErrchk(cudaMemcpy(
                l1__dev,
                hdr__file->l1,
                hdr__file->first_block_base - sizeof(*hdr),
                cudaMemcpyHostToDevice));

    free(hdr__file);

    //dev_fd_t ret;
    //ret.l1__dev = l1__dev;
    //ret.block_bit = hdr->block_bit;
    //ret.nr_l1 = hdr->nr_l1;

    PageCache *pagecache_dev = host_open_geminifs_file_for_device_1(pagecache_capacity,
            1 << hdr->block_bit,
            false,
            hdr->virtual_space_size);
    return pagecache_dev;
}

dev_fd_t
host_open_geminifs_file_for_device_without_backing_file(int page_size, uint64_t pagecache_capacity) {
    PageCache *pagecache_dev = host_open_geminifs_file_for_device_1(pagecache_capacity,
            page_size,
            true,
            pagecache_capacity);
    return pagecache_dev;
}

__global__ void
device_xfer_geminifs_file(dev_fd_t fd,
                          vaddr_t va,
                          void *buf_dev_1,
                          size_t nbyte,
                          int is_read) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint8_t *buf_dev = (uint8_t *)buf_dev_1;

    PageCache *pagecache = (PageCache *)fd;
    int page_bit_num = pagecache->get_page_bit_num();

#define PAGE_SIZE(pg_bit) (1 << (pg_bit))
#define PAGE_ID(vaddr, pg_bit) ((vaddr) >> (pg_bit))
#define PAGE_BASE(vaddr, pg_bit) ((vaddr) & ~(PAGE_SIZE(pg_bit) - 1))
#define PAGE_BASE__BY_ID(pg_id, pg_bit) ((pg_id) << (pg_bit))
#define PAGE_OFST(vaddr, pg_bit) ((vaddr) & (PAGE_SIZE(pg_bit) - 1))

    FilePageId begin = PAGE_ID(va, page_bit_num);
    FilePageId exclusive_end = PAGE_ID(va + nbyte, page_bit_num);
    FilePageId inclusive_end = exclusive_end - 1;
    if (begin == inclusive_end) {
        // not across the boundary of pages and in one page copy
        if (tid == 0) {
            CachePageId cachepage_id = pagecache->acquire_page(begin);
            uint8_t *cachepage_base = pagecache->get_raw_page_buf(cachepage_id);
            if (is_read) {
                memcpy(buf_dev, cachepage_base + PAGE_OFST(va, page_bit_num), nbyte);
            } else {
                memcpy(cachepage_base + PAGE_OFST(va, page_bit_num), buf_dev, nbyte);
                pagecache->set_page_dirty(cachepage_id);
            }
            pagecache->release_page(begin);
        }
        return;
    } else {
        // across the boundary of pages, and we deal non-full pages
        if (PAGE_OFST(va, page_bit_num) != 0) {
            // the first non-full page
            size_t n = PAGE_BASE__BY_ID(PAGE_ID(va, page_bit_num) + 1, page_bit_num) - va;
            if (tid == 0) {
                CachePageId cachepage_id = pagecache->acquire_page(begin);
                uint8_t *cachepage_base = pagecache->get_raw_page_buf(cachepage_id);
                if (is_read) {
                    memcpy(buf_dev, cachepage_base + PAGE_OFST(va, page_bit_num), n);
                } else {
                    memcpy(cachepage_base + PAGE_OFST(va, page_bit_num), buf_dev, n);
                    pagecache->set_page_dirty(cachepage_id);
                }
                pagecache->release_page(begin);
            }
            va += n;
            buf_dev += n;
            nbyte -= n;
        }

        if (PAGE_OFST(va + nbyte, page_bit_num) != 0) {
            // the last non-full page
            size_t n = (va + nbyte) - PAGE_BASE(va + nbyte, page_bit_num);
            if (tid == 0) {
                CachePageId cachepage_id = pagecache->acquire_page(inclusive_end);
                uint8_t *cachepage_base = pagecache->get_raw_page_buf(cachepage_id);
                uint8_t *dist_start = buf_dev + (nbyte - n);
                if (is_read) {
                    memcpy(dist_start, cachepage_base, n);
                } else {
                    memcpy(cachepage_base, dist_start, n);
                    pagecache->set_page_dirty(cachepage_id);
                }
                pagecache->release_page(inclusive_end);
            }
            nbyte -= n;
        }

    }

    assert(PAGE_OFST(va, page_bit_num) == 0);
    assert(PAGE_OFST(va + nbyte, page_bit_num) == 0);

    begin = PAGE_ID(va, page_bit_num);
    exclusive_end = PAGE_ID(va + nbyte, page_bit_num);

    size_t nr_block = gridDim.x;
    size_t block_id = blockIdx.x;
    size_t nr_thread_per_block = blockDim.x;
    size_t nr_warp__per_block = nr_thread_per_block / 32;
    size_t warp_id__in_block = threadIdx.x / 32;

    if (nr_warp__per_block == 0)
        // only one warp holding less 32 threads
        nr_warp__per_block++;

    if (nr_warp__per_block <= warp_id__in_block)
        // drop less-32-thread warp
        return;

    size_t nr_thread_in_the_warp = __popc(__activemask());
    assert(__popc(nr_thread_in_the_warp) == 1); // is power of 2

    size_t nr_page = exclusive_end - begin;
    size_t nr_page__per_block = nr_page / nr_block;
    if (nr_page % nr_block != 0)
        nr_page__per_block++;

    FilePageId begin__block = block_id * nr_page__per_block;
    FilePageId exclusive_end__block = begin__block + nr_page__per_block;
    if (exclusive_end <= begin__block)
        return;

    if (exclusive_end < exclusive_end__block)
        exclusive_end__block = exclusive_end;

    size_t nr_page__block = exclusive_end__block - begin__block;
    size_t nr_page__per_warp = nr_page__block / nr_warp__per_block;
    if (nr_page__block % nr_warp__per_block != 0)
        nr_page__per_warp++;

    FilePageId begin__warp = warp_id__in_block * nr_page__per_warp;
    FilePageId exclusive_end__warp = begin__warp + nr_page__per_warp;
    if (exclusive_end__block <= begin__warp)
        return;

    if (exclusive_end__block < exclusive_end__warp)
        exclusive_end__warp = exclusive_end__block;

    uint32_t participating_mask = __activemask();
    participating_mask = __match_any_sync(participating_mask, begin__warp);
    int page_size = PAGE_SIZE(page_bit_num);

    buf_dev = buf_dev + PAGE_BASE__BY_ID(begin__warp - begin, page_bit_num);
    for (FilePageId filepage_id = begin__warp;
            filepage_id < exclusive_end__warp;
            filepage_id++) {
        CachePageId cachepage_id = pagecache->acquire_page(inclusive_end);
        uint8_t *cachepage_base = pagecache->get_raw_page_buf(cachepage_id);
        if (is_read) {
            uint8_t *raw_page = cachepage_base;
            uint8_t *buf_dev_1 = buf_dev;
            for (size_t i = 0; i < page_size / 4096; i++) {
                warp_memcpy_4kB(buf_dev_1, raw_page, participating_mask);
                buf_dev_1 += 4096;
                raw_page += 4096;
            }
        } else {
            uint8_t *raw_page = cachepage_base;
            uint8_t *buf_dev_1 = buf_dev;
            for (size_t i = 0; i < page_size / 4096; i++) {
                warp_memcpy_4kB(raw_page, buf_dev_1, participating_mask);
                buf_dev_1 += 4096;
                raw_page += 4096;
            }
            pagecache->set_page_dirty(cachepage_id);
        }
        pagecache->release_page(inclusive_end);
        buf_dev += PAGE_SIZE(page_bit_num);
    }
}

int
main() {
  int block_size = 1ull << 12; /* 4096 */
  size_t capacity = 2 * (1ull << 30); /* 2G */
  dev_fd_t dev_fd =
      host_open_geminifs_file_for_device_without_backing_file(block_size, capacity);


  size_t buf_size = 512 * (1ull << 20); /* 512M */

  uint32_t *host_buf = (uint32_t *)malloc(buf_size);
  uint32_t *dev_buf1;
  uint32_t *dev_buf2;
  gpuErrchk(cudaMalloc(&dev_buf1, buf_size));
  gpuErrchk(cudaMalloc(&dev_buf2, buf_size));

  for (size_t i = 0; i < buf_size / sizeof(uint32_t); i++)
    host_buf[i] = i + 2;

  gpuErrchk(cudaMemcpy(dev_buf1, host_buf, buf_size, cudaMemcpyHostToDevice));

  for (vaddr_t va = 0; va < capacity; va += buf_size) {
      device_xfer_geminifs_file<<<108, 32>>>(dev_fd, va, dev_buf1, buf_size, 0);
  }

  for (vaddr_t va = 0; va < capacity; va += buf_size) {
      device_xfer_geminifs_file<<<108, 32>>>(dev_fd, va, dev_buf2, buf_size, 1);
  }

  gpuErrchk(cudaMemcpy(host_buf, dev_buf2, buf_size, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < buf_size / sizeof(uint32_t); i++)
    assert(host_buf[i] == i + 2);

  return 0;
}


//__device__ static nvme_ofst_t
//device__convert_va__to(dev_fd_t dev_fd,
//                       vaddr_t va) {
//  auto nr_l1 = dev_fd.nr_l1;
//  auto block_bit = dev_fd.block_bit;
//
//  uint64_t l1_idx = va >> block_bit;
//
//  if (l1_idx < nr_l1)
//    return dev_fd.l1__dev[l1_idx];
//
//  return 0;
//}
//
//__global__ void
//device_convert_va(dev_fd_t dev_fd,
//                  vaddr_t va,
//                  nvme_ofst_t *ret__dev) {
//  *ret__dev = device__convert_va__to(dev_fd, va);
//}
//  
//nvme_ofst_t
//host_convert_va__using_device(dev_fd_t dev_fd,
//        vaddr_t va) {
//  nvme_ofst_t *ret__dev;
//  assert(cudaSuccess ==
//    cudaMalloc(&ret__dev, sizeof(nvme_ofst_t))
//  );
//
//  device_convert_va<<<1, 1>>>(dev_fd, va, ret__dev);
//
//  nvme_ofst_t ret;
//  assert(cudaSuccess ==
//    cudaMemcpy(
//      &ret,
//      ret__dev,
//      sizeof(nvme_ofst_t),
//      cudaMemcpyDeviceToHost)
//  );
//
//  assert(cudaSuccess == cudaFree(ret__dev));
//  return ret;
//}
