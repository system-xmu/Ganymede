//#include <stdio.h>
//#include <assert.h>
//#include <sys/types.h>
//#include <unistd.h>

#include <cuda/atomic>
#include <cuda/semaphore>
#include <cuco/static_map.cuh>
#include <cuco/static_set.cuh>

#include <cooperative_groups.h>

#include "geminifs_api.h"
#include "geminifs_api.cuh"
#include "geminifs_internal.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
static __device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
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


class CachePage_WithoutBacking: public CachePage {
public:
    __device__
    CachePage_WithoutBacking(void * const buf_): CachePage(buf_, 1) { }
private:
    __device__ void __write_back(FilePageId filepage_id, void *info1, void *info2) { }
    __device__ void __read_in(FilePageId filepage_id, void *info1, void *info2) { }
};


class PageCacheImpl__info1 {
public:
    FilePageId filepage_id;
    cuda::counting_semaphore<cuda::thread_scope_device> wait_for_evicting;
    int nr_waiting;
};

template <typename Map1_Ptr, typename Map1_DevRef,
          typename Map2_Ptr, typename Map2_DevRef,
          typename Map3_Ptr, typename Map3_DevRef>
class PageCacheImpl: public PageCache {
public:
    __device__ PageCacheImpl(uint64_t pagecache_capacity,
            int page_size,
            uint64_t size_of_virtual_space,
            CachePage *pages[],
            void *info1, void *info2,
            Map1_Ptr map1, Map1_DevRef map1_ref,
            Map2_Ptr map2, Map2_DevRef map2_ref,
            Map3_Ptr map3, Map3_DevRef map3_ref):
        filepages__waiting_for_evicting(map1),
        filepages__waiting_for_evicting__ref(map1_ref),
        nr_waiting(0),
        filepage__to__cachepage(map2),
        filepage__to__cachepage__ref(map2_ref),
        zero_reffed_filepages(map3),
        zero_reffed_filepages__ref(map3_ref) {

        this->info1 = info1;
        this->info2 = info2;

        this->pagecache_lock.release();

        uint64_t nr_page = pagecache_capacity / page_size;

        this->page_size = page_size;
        this->nr_page = nr_page;

        this->pages = pages;
        this->pages_ref = (uint64_t *)malloc(sizeof(uint64_t) * this->nr_page);

        for (CachePageId cachepage_id = 0; cachepage_id < nr_page; cachepage_id++) {
            FilePageId filepage_id = cachepage_id;
            auto *page = this->pages[cachepage_id];
            page->cachepage_id = cachepage_id;
            page->content_of = filepage_id;
            page->assigned_to = filepage_id;
            page->lock.release();
            this->__insert__filepage__mapping_to__cachepage(filepage_id, cachepage_id);
            printf("%llx\n", cachepage_id);
            if (page->never_evicted) {
                page->state = CACHEPAGE_CLEAN;
                this->pages_ref[cachepage_id] = 1; // the Ref Count is at lease 1, thus the page won't be evicted.
            } else {
                page->state = CACHEPAGE_INVALID;
                this->pages_ref[cachepage_id] = 0;
                this->__insert__zero_reffed_filepage(filepage_id);
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

    __device__ CachePageId
    acquire_page(FilePageId filepage_id, uint32_t participating_mask) {
        CachePageId cachepage_id;

        uint32_t mask = participating_mask;
        mask = __match_any_sync(mask, (uint64_t)this);
        mask = __match_any_sync(mask, filepage_id);
        uint32_t warp_leader = __ffs(mask) - 1;
        int lane = my_lane_id();
        if (lane == warp_leader) {
            printf("acquire_page %llx\n", filepage_id);
            cachepage_id = this->__acquire_page_for_warp_leader(filepage_id);
        }
        cachepage_id = __shfl_sync(mask, cachepage_id, warp_leader);
        return cachepage_id;
    }

    __device__ void
    set_page_dirty(CachePageId cachepage_id) {
        uint32_t mask = __activemask();
        mask = __match_any_sync(mask, (uint64_t)this);
        mask = __match_any_sync(mask, cachepage_id);
        uint32_t warp_leader = __ffs(mask) - 1;
        int lane = my_lane_id();
        if (lane == warp_leader)
            this->pages[cachepage_id]->set_dirty();
    }

    __device__ void
    release_page(FilePageId filepage_id, uint32_t participating_mask) {
        uint32_t mask = participating_mask;
        mask = __match_any_sync(mask, (uint64_t)this);
        mask = __match_any_sync(mask, filepage_id);
        uint32_t warp_leader = __ffs(mask) - 1;
        int lane = my_lane_id();
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

        printf("I'm in. I want %llx!!\n", filepage_id);
        ret = this->__get__cachepage_id(filepage_id);
        if (ret != -1) {
            // Page Hit!
        printf("Hit!!\n");
            size_t cur_ref = (++(this->pages_ref[ret]));
            if (cur_ref == 1)
                this->__erase__zero_reffed_filepage(filepage_id);

            __threadfence();
            this->pagecache_lock.release();

        printf("I release the lock\n");
            this->pages[ret]->lock.acquire();
            this->pages[ret]->read_in__no_lock(this->info1, this->info2);
            this->pages[ret]->lock.release();

        //printf("I leave~\n");
            return ret;
        }

        assert(0);
        printf("Miss!!\n");

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
            this->pages[ret]->read_in__no_lock(this->info1, this->info2);
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
            assert(leaders_waiting_for_evicting);
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
        this->pages[ret]->read_in__no_lock(this->info1, this->info2);
        this->pages[ret]->lock.release();
        return ret;
    }

    __forceinline__ __device__ void
    __release_page_for_warp_leader(FilePageId filepage_id) {
        this->pagecache_lock.acquire();
        printf("I'm %llx, I want to release the page %llx\n", (uint64_t)my_lane_id(), filepage_id);
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
                this->pages[cachepage_id]->write_back__no_lock(this->info1, this->info2);
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
        auto found = this->filepages__waiting_for_evicting__ref.find(filepage_id);
        if (found != this->filepages__waiting_for_evicting__ref.end()) {
            auto *n = static_cast<MyLinklistNodeD<PageCacheImpl__info1 *> *>(found->second);
            return n->v;
        } else
            return nullptr;
    }

    __forceinline__ __device__ bool
    __has_quota_to_wait_for_evicting() {
        return this->nr_waiting < this->filepages__waiting_for_evicting__ref.capacity() / 2;
    }

    __forceinline__ __device__ void
    __insert__filepage_waiting_for_evicting(FilePageId filepage_id,
                                             PageCacheImpl__info1 *p) {
        auto *n = new MyLinklistNodeD<PageCacheImpl__info1 *>();
        assert(n);
        n->v = p;
        this->filepages__waiting_for_evicting__list.enqueue(n);
        this->nr_waiting++;

        assert(this->filepages__waiting_for_evicting__ref.insert(cuco::pair{filepage_id, n}));
    }

    __forceinline__ __device__ PageCacheImpl__info1 *
    __pop__filepage_waiting_for_evicting() {
        // get and erase
        auto *n = static_cast<MyLinklistNodeD<PageCacheImpl__info1 *> *>(this->filepages__waiting_for_evicting__list.pop());
        auto ret = n->v;
        this->nr_waiting--;

        assert(this->filepages__waiting_for_evicting__ref.erase(ret->filepage_id));

        return ret;
    }
//------------------------------------------------------------
    __forceinline__ __device__ void
    __insert__filepage__mapping_to__cachepage(FilePageId filepage_id,
                                              CachePageId cachepage_id) {
        assert(this->filepage__to__cachepage__ref.insert(cuco::pair{filepage_id, cachepage_id}));
    }

    __forceinline__ __device__ void
    __erase__filepage__mapping(FilePageId filepage_id) {
        assert(this->filepage__to__cachepage__ref.erase(filepage_id));
    }

    __forceinline__ __device__ CachePageId
    __get__cachepage_id(FilePageId filepage_id) {
        auto found = this->filepage__to__cachepage__ref.find(filepage_id);
        if (found != this->filepage__to__cachepage__ref.end())
            return found->second;
        else
            return -1;
    }
//-----------------------------------------------------------
    __forceinline__ __device__ void
    __insert__zero_reffed_filepage(FilePageId filepage_id) {
        auto *n = new MyLinklistNodeD<FilePageId>();
        assert(n);
        n->v = filepage_id;
        this->zero_reffed_filepages__list.enqueue(n);

        assert(this->zero_reffed_filepages__ref.insert(cuco::pair{filepage_id, n}));
    }

    __forceinline__ __device__ void
    __erase__zero_reffed_filepage(FilePageId filepage_id) {
        MyLinklistNodeD<FilePageId> *n = nullptr;
        auto found = this->zero_reffed_filepages__ref.find(filepage_id);
        if (found != this->zero_reffed_filepages__ref.end())
            n = static_cast<MyLinklistNodeD<FilePageId> *>(found->second);

        assert(n);

        assert(this->zero_reffed_filepages__ref.erase(filepage_id));
        this->zero_reffed_filepages__list.detach_node(n);
        delete n;
    }

    __forceinline__ __device__ FilePageId
    __pop__zero_reffed_filepage_id() {
        // get and erase
        auto *n = static_cast<MyLinklistNodeD<FilePageId> *>(this->zero_reffed_filepages__list.pop());
        auto filepage_id = n->v;
        delete n;

        assert(this->zero_reffed_filepages__ref.erase(filepage_id));

        return filepage_id;
    }
//-----------------------------------------------------------

    void *info1, *info2;

    cuda::binary_semaphore<cuda::thread_scope_device> pagecache_lock;

    void *nvme_controller;

    int page_size;

    uint64_t nr_page;

    CachePage **pages;
    uint64_t *pages_ref;

    size_t nr_waiting;
    Map1_Ptr filepages__waiting_for_evicting;
    Map1_DevRef filepages__waiting_for_evicting__ref;
    MyLinklist filepages__waiting_for_evicting__list;

    Map2_Ptr filepage__to__cachepage;
    Map2_DevRef filepage__to__cachepage__ref;

    Map3_Ptr zero_reffed_filepages;
    Map3_DevRef zero_reffed_filepages__ref;
    MyLinklist zero_reffed_filepages__list;
};

__host__ PageCache *
__internal__get_pagecache(
        uint64_t pagecache_capacity,
        int page_size,
        uint64_t size_of_virtual_space,
        CachePage *pages[],
        void *info1, void *info2) {
    uint64_t nr_page = pagecache_capacity / page_size;

    // Initial host part of containers
    FilePageId constexpr empty_FilePageId_sentinel = -1;
    CachePageId constexpr empty_CachePageId_sentinel = -1;
    MyLinklistNode constexpr *sentinel = nullptr;

    auto *filepages__waiting_for_evicting =
        new cuco::static_map
        <FilePageId,
        MyLinklistNode *,
        cuco::extent<std::size_t>,
        cuda::thread_scope_device,
        thrust::equal_to<FilePageId>,
        cuco::linear_probing<1, cuco::default_hash_function<FilePageId>>>
            (2 * nr_page,
             cuco::empty_key{empty_FilePageId_sentinel},
             cuco::empty_value{sentinel});
    auto map1_ref = filepages__waiting_for_evicting->ref(cuco::insert, cuco::find, cuco::erase);

    auto *filepage__to__cachepage =
        new cuco::static_map
        <FilePageId,
        CachePageId,
        cuco::extent<std::size_t>,
        cuda::thread_scope_device,
        thrust::equal_to<FilePageId>,
        cuco::linear_probing<1, cuco::default_hash_function<FilePageId>>>
            (2 * nr_page,
             cuco::empty_key{empty_FilePageId_sentinel},
             cuco::empty_value{empty_CachePageId_sentinel});
    auto map2_ref = filepage__to__cachepage->ref(cuco::insert, cuco::find, cuco::erase);

    auto *zero_reffed_filepages =
        new cuco::static_map
        <FilePageId,
        MyLinklistNode *,
        cuco::extent<std::size_t>,
        cuda::thread_scope_device,
        thrust::equal_to<FilePageId>,
        cuco::linear_probing<1, cuco::default_hash_function<FilePageId>>>
            (2 * nr_page,
             cuco::empty_key{empty_FilePageId_sentinel},
             cuco::empty_value{sentinel});
    auto map3_ref = zero_reffed_filepages->ref(cuco::insert, cuco::find, cuco::erase);

    using PageCacheImplType =
        PageCacheImpl<decltype(filepages__waiting_for_evicting), decltype(map1_ref),
                      decltype(filepage__to__cachepage), decltype(map2_ref),
                      decltype(zero_reffed_filepages), decltype(map3_ref)>;

    PageCache *pagecache;
    //gpuErrchk(cudaMallocManaged(&pagecache, sizeof(PageCacheImplType)));
    gpuErrchk(cudaMalloc(&pagecache, sizeof(PageCacheImplType)));

    RUN_ON_DEVICE({
        new (pagecache) PageCacheImplType (pagecache_capacity,
                page_size,
                size_of_virtual_space,
                pages,
                info1, info2,
                filepages__waiting_for_evicting, map1_ref,
                filepage__to__cachepage, map2_ref,
                zero_reffed_filepages, map3_ref);
    });
    return pagecache;
}

dev_fd_t
host_open_geminifs_file_for_device_without_backing_file(int page_size, uint64_t pagecache_capacity) {
    uint64_t nr_page = pagecache_capacity / page_size;

    uint8_t *all_raw_pages;
    gpuErrchk(cudaMalloc(&all_raw_pages, nr_page * page_size));

    CachePage_WithoutBacking *cachepage_structures;
    gpuErrchk(cudaMalloc(&cachepage_structures, sizeof(CachePage_WithoutBacking) * nr_page));

    CachePage **pages;
    gpuErrchk(cudaMalloc(&pages, sizeof(CachePage *) * nr_page));

    RUN_ON_DEVICE({
        for (size_t i = 0; i < nr_page; i++) {
            auto *cachepage = cachepage_structures + i;
            pages[i] = new (cachepage) CachePage_WithoutBacking(all_raw_pages + i * page_size);
        }
    });

    PageCache *pagecache_dev = __internal__get_pagecache(pagecache_capacity,
            page_size,
            pagecache_capacity,
            pages,
            nullptr, nullptr);
    return pagecache_dev;
}

__global__ void
device_xfer_geminifs_file(dev_fd_t fd,
                          vaddr_t va,
                          void *buf_dev_1,
                          size_t nbyte,
                          int is_read) {
    size_t nr_block = gridDim.x;
    size_t block_id = blockIdx.x;
    size_t nr_thread_per_block = blockDim.x;
    size_t nr_warp__per_block = nr_thread_per_block / 32;
    size_t warp_id__in_block = threadIdx.x / 32;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    assert(0 < nr_warp__per_block); // Every warp must hold 32 threads

    if (nr_warp__per_block <= warp_id__in_block)
        // drop less-32-thread warp
        return;

    __syncwarp();
    size_t nr_thread_in_the_warp = __popc(__activemask());
    assert(nr_thread_in_the_warp == 32);




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
            uint32_t participating_mask = __activemask();
            CachePageId cachepage_id = pagecache->acquire_page(begin, participating_mask);
            uint8_t *cachepage_base = pagecache->get_raw_page_buf(cachepage_id);
            if (is_read) {
                memcpy(buf_dev, cachepage_base + PAGE_OFST(va, page_bit_num), nbyte);
            } else {
                memcpy(cachepage_base + PAGE_OFST(va, page_bit_num), buf_dev, nbyte);
                pagecache->set_page_dirty(cachepage_id);
            }
            pagecache->release_page(begin, participating_mask);
        }
        return;
    } else {
        // across the boundary of pages, and we deal non-full pages
        if (PAGE_OFST(va, page_bit_num) != 0) {
            // the first non-full page
            size_t n = PAGE_BASE__BY_ID(PAGE_ID(va, page_bit_num) + 1, page_bit_num) - va;
            if (tid == 0) {
                uint32_t participating_mask = __activemask();
                CachePageId cachepage_id = pagecache->acquire_page(begin, participating_mask);
                uint8_t *cachepage_base = pagecache->get_raw_page_buf(cachepage_id);
                if (is_read) {
                    memcpy(buf_dev, cachepage_base + PAGE_OFST(va, page_bit_num), n);
                } else {
                    memcpy(cachepage_base + PAGE_OFST(va, page_bit_num), buf_dev, n);
                    pagecache->set_page_dirty(cachepage_id);
                }
                pagecache->release_page(begin, participating_mask);
            }
            va += n;
            buf_dev += n;
            nbyte -= n;
        }

        if (PAGE_OFST(va + nbyte, page_bit_num) != 0) {
            // the last non-full page
            size_t n = (va + nbyte) - PAGE_BASE(va + nbyte, page_bit_num);
            if (tid == 0) {
                uint32_t participating_mask = __activemask();
                CachePageId cachepage_id = pagecache->acquire_page(inclusive_end, participating_mask);
                uint8_t *cachepage_base = pagecache->get_raw_page_buf(cachepage_id);
                uint8_t *dist_start = buf_dev + (nbyte - n);
                if (is_read) {
                    memcpy(dist_start, cachepage_base, n);
                } else {
                    memcpy(cachepage_base, dist_start, n);
                    pagecache->set_page_dirty(cachepage_id);
                }
                pagecache->release_page(inclusive_end, participating_mask);
            }
            nbyte -= n;
        }

    }

    assert(PAGE_OFST(va, page_bit_num) == 0);
    assert(PAGE_OFST(va + nbyte, page_bit_num) == 0);

    begin = PAGE_ID(va, page_bit_num);
    exclusive_end = PAGE_ID(va + nbyte, page_bit_num);


    size_t nr_page = exclusive_end - begin;
    size_t nr_page__per_block = nr_page / nr_block;
    if (nr_page % nr_block != 0)
        nr_page__per_block++;

    FilePageId begin__block = begin + block_id * nr_page__per_block;
    FilePageId exclusive_end__block = begin__block + nr_page__per_block;
    if (exclusive_end <= begin__block)
        return;

    if (exclusive_end < exclusive_end__block)
        exclusive_end__block = exclusive_end;

    size_t nr_page__block = exclusive_end__block - begin__block;
    size_t nr_page__per_warp = nr_page__block / nr_warp__per_block;
    if (nr_page__block % nr_warp__per_block != 0)
        nr_page__per_warp++;

    FilePageId begin__warp = begin__block + warp_id__in_block * nr_page__per_warp;
    FilePageId exclusive_end__warp = begin__warp + nr_page__per_warp;
    if (exclusive_end__block <= begin__warp)
        return;

    if (exclusive_end__block < exclusive_end__warp)
        exclusive_end__warp = exclusive_end__block;

    __syncwarp();
    uint32_t participating_mask = __activemask();
    participating_mask = __match_any_sync(participating_mask, begin__warp);
    int page_size = PAGE_SIZE(page_bit_num);


    size_t warp_id__overview = warp_id__in_block + nr_warp__per_block * block_id;
    if (0 == my_lane_id()) {
    printf("I'm warp %llx (in-block id %llx) threadIdx.x %llx, I account for [%llx, %llx)\n",
            warp_id__overview, warp_id__in_block, threadIdx.x, begin__warp, exclusive_end__warp);
    }

    buf_dev = buf_dev + PAGE_BASE__BY_ID(begin__warp - begin, page_bit_num);
    for (FilePageId filepage_id = begin__warp;
            filepage_id < exclusive_end__warp;
            filepage_id++) {
        CachePageId cachepage_id = pagecache->acquire_page(filepage_id, participating_mask);
        uint8_t *cachepage_base = pagecache->get_raw_page_buf(cachepage_id);
        __syncwarp(participating_mask);
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
        pagecache->release_page(filepage_id, participating_mask);
        buf_dev += PAGE_SIZE(page_bit_num);
    }
}

