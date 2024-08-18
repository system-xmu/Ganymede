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
    __write_back(FilePageId filepage_id, void *ctrl, void *hdr) {
        this->__xfer(filepage_id, ctrl, hdr, 1);
    }

    __device__ void
    __read_in(FilePageId filepage_id, void *ctrl, void *hdr) {
        this->__xfer(filepage_id, ctrl, hdr, 0);
    }

    __device__ __forceinline__ void
    __xfer(FilePageId filepage_id, void *ctrl_, void *hdr_, int is_read) {
        auto *ctrl = (Controller *)ctrl_;
        auto *hdr = (struct geminiFS_hdr *)hdr_;


        int page_size = this->page_size;
        int page_bit = __popc(page_size - 1);
        int file_block_size = 1 << hdr->block_bit;



        if (file_block_size < (128 * (1ull << 10))) {
            vaddr_t file_va = filepage_id << page_bit;
            nvme_ofst_t nvme_ofst = this->__get_nvmeofst(hdr, file_va);

            int queue = 0;
            QueuePair* qp = &ctrl->d_qps[queue];

            size_t start_hqps_block = nvme_ofst >> this->hqps_block_size_log;
            int nr_hqps_blocks = file_block_size >> this->hqps_block_size_log;
            nvm_cmd_t cmd;
            uint16_t cid = get_cid(&(qp->sq));
            nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
            
            assert(this->n_ioaddrs == 1);
            uint64_t prp1 = this->ioaddrs[0];
            uint64_t prp2 = 0;
            nvm_cmd_data_ptr(&cmd, prp1, prp2);
            nvm_cmd_rw_blks(&cmd, start_hqps_block, nr_hqps_blocks);
            uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
            uint32_t head, head_;

            uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);

            qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);

            cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);

            put_cid(&qp->sq, cid);
        } else {
            assert(0);
// page_size
// |--------------------------------------------------|

// file_block_size == nvme_block_size
// |------------|

// n_ioaddrs (here is 16)
// |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

            int nr_file_blocks__in_page = page_size / file_block_size;
            int nr_hqps_blocks = file_block_size >> this->hqps_block_size_log;

            int nvme_block_size = file_block_size;
            int nr_ioaddrs__per_nvme_block = this->n_ioaddrs / nr_file_blocks__in_page;

            vaddr_t file_page_va = filepage_id << page_bit;

            int queue = 0;
            QueuePair* qp = &ctrl->d_qps[queue];

            for (size_t i = 0; i < nr_file_blocks__in_page; i++) {
                vaddr_t file_block_va = file_page_va + i * file_block_size;
                nvme_ofst_t nvme_ofst = this->__get_nvmeofst(hdr, file_block_va);
                size_t start_hqps_block = nvme_ofst >> this->hqps_block_size_log;
                for (size_t j = 0; j < nr_ioaddrs__per_nvme_block; j++) {
                    int ioaddr_idx = i * nr_ioaddrs__per_nvme_block + j;
                }
            }

            nvm_cmd_t cmd;
            uint16_t cid = get_cid(&(qp->sq));
            nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
            //uint64_t prp1 = dev_buf[0];
            //uint64_t prp2 = 0;
            //nvm_cmd_data_ptr(&cmd, prp1, prp2);
            //nvm_cmd_rw_blks(&cmd, start_block, n_blocks);
            //uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
            //uint32_t head, head_;

            //uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);

            //qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);

            //cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);

            //put_cid(&qp->sq, cid);
        }



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

    RUN_ON_DEVICE({
        for (size_t i = 0; i < nr_page; i++) {
            auto *cachepage = cachepage_structures + i;
            pages[i] = new (cachepage) CachePage_NvmeBacking(page_size);
        }
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
            ctrl->n_qps,
            ctrl->d_ctrl_ptr, hdr__dev);
}
