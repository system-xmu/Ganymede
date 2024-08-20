#include "geminifs_api.cuh"
#include "geminifs_internal.cuh"

#include <ctrl.h>
#include <buffer.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_types.h>
#include <nvm_util.h>
#include <nvm_cmd.h>
#include "get-offset/get-offset.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

class QueueAcquireHelper {
public:
    __device__
    QueueAcquireHelper(int nr_queues) {
        this->nr_queues = nr_queues;
        this->locks = new cuda::binary_semaphore<cuda::thread_scope_device> [nr_queues];
        for (size_t i = 0; i < nr_queues; i++)
            this->locks[i].release();
    }

    __forceinline__ __device__ int
    acquire_queue() {
        //int queue = get_smid() % this->nr_queues;
        int queue = 0;
        this->locks[queue].acquire();
        return queue;
    }

    __forceinline__ __device__ void
    release_queue(int queue) {
        this->locks[queue].release();
    }

private:
    int nr_queues;
    cuda::binary_semaphore<cuda::thread_scope_device> *locks;
};

class CachePage_NvmeBacking: public CachePage {
public:
    __device__
    CachePage_NvmeBacking(int page_size): CachePage(page_size, nullptr, 0) { }

    DmaPtr gpu_buffer;

    int n_ioaddrs;
    uint64_t *ioaddrs;
    int hqps_block_size_log;
private:
    __device__ __forceinline__ nvme_ofst_t
    __get_nvmeofst(struct geminiFS_hdr *hdr, vaddr_t va) {
        uint64_t l1_idx = va >> hdr->block_bit;
        if (l1_idx < hdr->nr_l1)
            return hdr->l1[l1_idx];
        return 0;
    }

    __device__ void
    __write_back(FilePageId filepage_id, void *ctrl, void *hdr, void *queue_acquire_helper) {
        this->__xfer(filepage_id, ctrl, hdr, queue_acquire_helper, 0);
    }

    __device__ void
    __read_in(FilePageId filepage_id, void *ctrl, void *hdr, void *queue_acquire_helper) {
        this->__xfer(filepage_id, ctrl, hdr, queue_acquire_helper, 1);
    }

    __device__ __forceinline__ void
    __xfer(FilePageId filepage_id, void *ctrl_, void *hdr_, void *queue_acquire_helper_, int is_read) {
        auto *ctrl = (Controller *)ctrl_;
        auto *hdr = (struct geminiFS_hdr *)hdr_;
        auto *queue_acquire_helper = (QueueAcquireHelper *)queue_acquire_helper_;


        int page_size = this->page_size;
        int page_bit = __popc(page_size - 1);
        int file_block_size = 1 << hdr->block_bit;



        assert(file_block_size == page_size);

        vaddr_t file_va = filepage_id << page_bit;
        nvme_ofst_t nvme_ofst = this->__get_nvmeofst(hdr, file_va);

        int queue = queue_acquire_helper->acquire_queue();
        printf("I get a queue [%llx]\n", (uint64_t)queue);
        QueuePair* qp = &ctrl->d_qps[queue];

        size_t start_hqps_block = nvme_ofst >> this->hqps_block_size_log;
        int nr_hqps_blocks = file_block_size >> this->hqps_block_size_log;

        int size_of_ioaddr = 4096;
        int nr_hqps_blocks__per_ioaddr = 4096 / (1 << this->hqps_block_size_log);

        for (size_t idx_ioaddr = 0;
                idx_ioaddr < page_size / size_of_ioaddr;
                idx_ioaddr++, start_hqps_block += nr_hqps_blocks__per_ioaddr) {
            nvm_cmd_t cmd;
            uint16_t cid = get_cid(&(qp->sq));
            uint64_t prp1 = this->ioaddrs[idx_ioaddr];
            uint64_t prp2 = 0;
            {
                if (is_read) {
                    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
                    printf("read in filepage_id[%llx] file_va[%llx] nvmeofst[%llx] ioaddr[%llx] hqps_block_size_log[%llx]\n", filepage_id, file_va, nvme_ofst, this->ioaddrs[idx_ioaddr], (uint64_t)this->hqps_block_size_log);
                } else {
                    nvm_cmd_header(&cmd, cid, NVM_IO_WRITE, qp->nvmNamespace);
                    printf("write back filepage_id[%llx] file_va[%llx] nvmeofst[%llx] ioaddr[%llx] hqps_block_size_log[%llx]\n", filepage_id, file_va, nvme_ofst, this->ioaddrs[idx_ioaddr], (uint64_t)this->hqps_block_size_log);
                }

                nvm_cmd_data_ptr(&cmd, prp1, prp2);
                nvm_cmd_rw_blks(&cmd, start_hqps_block, nr_hqps_blocks__per_ioaddr);
                uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
                uint32_t head, head_;
                uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
                qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
                cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
                put_cid(&qp->sq, cid);
            }
            if (!is_read) {
                    nvm_cmd_header(&cmd, cid, NVM_IO_FLUSH, qp->nvmNamespace);
                nvm_cmd_data_ptr(&cmd, prp1, prp2);
                nvm_cmd_rw_blks(&cmd, start_hqps_block, nr_hqps_blocks__per_ioaddr);
                uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
                uint32_t head, head_;
                uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
                qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
                cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
                put_cid(&qp->sq, cid);
            }
        }

        printf("I release the queue [%llx]\n", (uint64_t)queue);
        queue_acquire_helper->release_queue(queue);
    }
};

static Controller *ctrl;

void
host_open_all(
        const char *snvme_control_path,
        const char *snvme_path,
        const char *nvme_dev_path,
        const char *mount_path,
        uint32_t ns_id,
        uint64_t queueDepth,
        uint64_t numQueues) {
    int device;
    gpuErrchk(cudaGetDevice(&device));
    ctrl = new Controller(
            (char *)snvme_control_path,
            (char *)snvme_path,
            (char *)nvme_dev_path,
            (char *)mount_path,
            ns_id, device, queueDepth, numQueues);
}

void
host_close_all() {
    delete ctrl;
}

dev_fd_t
host_open_geminifs_file_for_device(
        host_fd_t host_fd,
        uint64_t pagecache_capacity,
        int page_size) {
    struct geminiFS_hdr *hdr = host_fd;


    int file_block_size = 1 << hdr->block_bit;
    if (file_block_size < (128 * (1ull << 10))) {
        assert(file_block_size == page_size);
    } else {
        assert(file_block_size <= page_size);
    }

    struct geminiFS_hdr *hdr__dev;
    gpuErrchk(cudaMallocManaged(&hdr__dev, hdr->first_block_base));
    assert((off_t)(-1) != lseek(hdr->fd, 0, SEEK_SET));
    assert(hdr->first_block_base == read(hdr->fd, hdr__dev, hdr->first_block_base));

    size_t nr_page = pagecache_capacity / page_size;

    CachePage_NvmeBacking *cachepage_structures;
    gpuErrchk(cudaMallocManaged(&cachepage_structures, sizeof(CachePage_NvmeBacking) * nr_page));

    CachePage **pages;
    gpuErrchk(cudaMalloc(&pages, sizeof(CachePage *) * nr_page));

    int nr_queues = ctrl->n_qps;
    QueueAcquireHelper *queue_acquire_helper;
    gpuErrchk(cudaMalloc(&queue_acquire_helper, sizeof(QueueAcquireHelper)));

    RUN_ON_DEVICE({
        for (size_t i = 0; i < nr_page; i++) {
            auto *cachepage = cachepage_structures + i;
            pages[i] = new (cachepage) CachePage_NvmeBacking (page_size);
        }
        new (queue_acquire_helper) QueueAcquireHelper (nr_queues);
    });

    int device;
    gpuErrchk(cudaGetDevice(&device));

    for (size_t i = 0; i < nr_page; i++) {
        auto *cachepage = cachepage_structures + i;
        cachepage->gpu_buffer = createDma(ctrl->ctrl, page_size, device);
        cachepage->buf = cachepage->gpu_buffer->vaddr;
        cachepage->n_ioaddrs = cachepage->gpu_buffer->n_ioaddrs;
        gpuErrchk(cudaMallocManaged(&(cachepage->ioaddrs),
                    sizeof(uint64_t) * cachepage->gpu_buffer->n_ioaddrs));
        for (size_t j = 0; j < cachepage->gpu_buffer->n_ioaddrs; j++)
            cachepage->ioaddrs[j] = cachepage->gpu_buffer->ioaddrs[j];
        cachepage->hqps_block_size_log = ctrl->h_qps[0]->block_size_log;
    }


    return __internal__get_pagecache(pagecache_capacity,
            page_size,
            hdr->virtual_space_size,
            pages,
            nr_queues,
            ctrl->d_ctrl_ptr, hdr__dev, queue_acquire_helper);
}

class CachePage_TestForPageCache: public CachePage {
public:
    __device__
    CachePage_TestForPageCache(int page_size): CachePage(page_size, nullptr, 0) { }
private:
    __device__ __forceinline__ nvme_ofst_t
    __get_nvmeofst(struct geminiFS_hdr *hdr, vaddr_t va) {
        return 0;
    }

    __device__ void
    __write_back(FilePageId filepage_id, void *ctrl, void *hdr, void *queue_acquire_helper) {
        this->__xfer(filepage_id, ctrl, hdr, queue_acquire_helper, 1);
    }

    __device__ void
    __read_in(FilePageId filepage_id, void *ctrl, void *hdr, void *queue_acquire_helper) {
        this->__xfer(filepage_id, ctrl, hdr, queue_acquire_helper, 0);
    }

    __device__ __forceinline__ void
    __xfer(FilePageId filepage_id, void *ctrl_, void *hdr_, void *queue_acquire_helper_, int is_read) {
        auto *queue_acquire_helper = (QueueAcquireHelper *)queue_acquire_helper_;


        int queue = queue_acquire_helper->acquire_queue();
        //__nanosleep(1000);
        queue_acquire_helper->release_queue(queue);

    }
};

dev_fd_t
host_get_pagecache__for_test_evicting(
        uint64_t fake_file_size,
        uint64_t pagecache_capacity,
        int page_size) {
    size_t nr_page = pagecache_capacity / page_size;

    uint8_t *all_raw_pages;
    gpuErrchk(cudaMalloc(&all_raw_pages, nr_page * page_size));

    CachePage_TestForPageCache *cachepage_structures;
    gpuErrchk(cudaMallocManaged(&cachepage_structures, sizeof(CachePage_TestForPageCache) * nr_page));

    CachePage **pages;
    gpuErrchk(cudaMalloc(&pages, sizeof(CachePage *) * nr_page));

    int nr_queues = 32;
    QueueAcquireHelper *queue_acquire_helper;
    gpuErrchk(cudaMalloc(&queue_acquire_helper, sizeof(QueueAcquireHelper)));

    RUN_ON_DEVICE({
        for (size_t i = 0; i < nr_page; i++) {
            auto *cachepage = cachepage_structures + i;
            pages[i] = new (cachepage) CachePage_TestForPageCache (page_size);
            cachepage->buf = all_raw_pages + i * page_size;
        }
        new (queue_acquire_helper) QueueAcquireHelper (nr_queues);
    });

    return __internal__get_pagecache(pagecache_capacity,
            page_size,
            fake_file_size,
            pages,
            nr_queues,
            nullptr, nullptr, queue_acquire_helper);
}
