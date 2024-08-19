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
    CachePage_NvmeBacking(): CachePage(nullptr, 0) { }

    DmaPtr gpu_buffer;
private:
    __device__ void
    __write_back(FilePageId filepage_id, void *ctrl, void *va_to_nvmeofst) {
    }

    __device__ void
    __read_in(FilePageId filepage_id, void *ctrl, void *va_to_nvmeofst) {
    }
};


static Controller *
__get_nvme_ctrl(
        const char *snvme_control_path,
        const char *snvme_path,
        const char *nvme_dev_path,
        const char *mount_path,
        uint32_t ns_id,
        int device,
        uint64_t queueDepth,
        uint64_t numQueues) {
    return new Controller(
            (char *)snvme_control_path,
            (char *)snvme_path,
            (char *)nvme_dev_path,
            (char *)mount_path,
            ns_id, device, queueDepth, numQueues);
}

dev_fd_t
host_open_geminifs_file_for_device(
        host_fd_t host_fd,
        uint64_t pagecache_capacity,
        const char *snvme_control_path,
        const char *snvme_path,
        const char *nvme_dev_path,
        const char *mount_path) {
    struct geminiFS_hdr *hdr = host_fd;

    nvme_ofst_t *l1__dev;
    gpuErrchk(cudaMalloc(&l1__dev, hdr->first_block_base));

    struct geminiFS_hdr *hdr__file = (struct geminiFS_hdr *)malloc(hdr->first_block_base);

    assert((off_t)(-1) !=
            lseek(hdr->fd, 0, SEEK_SET));
    assert(hdr->first_block_base ==
            read(hdr->fd, hdr__file, hdr->first_block_base));
    gpuErrchk(cudaMemcpy(
                l1__dev,
                hdr__file->l1,
                hdr__file->first_block_base - sizeof(*hdr),
                cudaMemcpyHostToDevice));

    free(hdr__file);


    int block_size = 1 << hdr->block_bit;
    size_t nr_page = pagecache_capacity / block_size;

    CachePage_NvmeBacking *cachepage_structures;
    gpuErrchk(cudaMallocManaged(&cachepage_structures, sizeof(CachePage_NvmeBacking) * nr_page));

    CachePage **pages;
    gpuErrchk(cudaMalloc(&pages, sizeof(CachePage *) * nr_page));

    RUN_ON_DEVICE({
        for (size_t i = 0; i < nr_page; i++) {
            auto *cachepage = cachepage_structures + i;
            pages[i] = new (cachepage) CachePage_NvmeBacking();
        }
    });

    int device;
    gpuErrchk(cudaGetDevice(&device));
    // snvme module install -> mount 
    Controller *ctrl = __get_nvme_ctrl(snvme_control_path,
            snvme_path,
            nvme_dev_path,
            mount_path,
            1, device, 1024, 32);

    for (size_t i = 0; i < nr_page; i++) {
        auto *cachepage = cachepage_structures + i;
        cachepage->gpu_buffer = createDma(ctrl->ctrl,
                NVM_PAGE_ALIGN(block_size, 1UL << 16),
                device);
        cachepage->buf = (void *)cachepage->gpu_buffer->ioaddrs[0];
    }


    //dev_fd_t ret;
    //ret.l1__dev = l1__dev;
    //ret.block_bit = hdr->block_bit;
    //ret.nr_l1 = hdr->nr_l1;

    return __internal__get_pagecache(pagecache_capacity,
            block_size,
            hdr->virtual_space_size,
            pages,
            ctrl, nullptr);
}
