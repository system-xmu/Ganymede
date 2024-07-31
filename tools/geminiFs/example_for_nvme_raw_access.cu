#include <iostream>
#include <cassert>
#include <cstddef>

#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>

#include <cuda_runtime.h>

#include "../get-offset/get-offset.h"
#include "geminifs.h"
#include <ctrl.h>

__device__ void
read_from_nvme(nvme_ofst_t nvme_ofst, void *dev_buf, uint64_t len, int queue) {
    printf("nvme_ofst: 0x%lx\n", nvme_ofst);
    int *buf = (int *)dev_buf;
    for (size_t i = 0; i < len / sizeof(int); i++)
        buf[i] = i;
}

__device__ void
write_to_nvme(nvme_ofst_t nvme_ofst, void *dev_buf, uint64_t len, int queue) {
}

__device__ void
sync_nvme(nvme_ofst_t nvme_ofst) {
}

__global__ void
read_from_nvme__using_device(nvme_ofst_t nvme_ofst, void *dev_buf, uint64_t len, int queue) {
    read_from_nvme(nvme_ofst, dev_buf, len, queue);
}

__global__ void
write_to_nvme__using_device(nvme_ofst_t nvme_ofst, void *dev_buf, uint64_t len, int queue) {
    read_from_nvme(nvme_ofst, dev_buf, len, queue);
}

__global__ void
sync_nvme__using_device(nvme_ofst_t nvme_ofst) {
    sync_nvme(nvme_ofst);
}

int
main () {
    std::cout << "halo" << std::endl;

    int block_size = 4096;

    const char *filename = "test_for_nvme_access.raw";
    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    assert(0 <= fd);
    assert(0 == ftruncate(fd, block_size));

    int *buf__host = (int *)malloc(block_size);
    for (size_t i = 0; i < sizeof(buf__host) / sizeof(int); i++)
        buf__host[i] = i;
    assert(block_size == write(fd, buf__host, block_size));

#define snvme_helper_path "/dev/snvme_helper"
    int snvme_helper_fd = open(snvme_helper_path, O_RDWR);
    if (snvme_helper_fd < 0) {
        perror("Failed to open snvme_helper_fd");
        assert(0);
    }
    struct nds_mapping mapping;
    mapping.file_fd = fd;
    mapping.offset = 0;
    mapping.len = block_size;
    if (ioctl(snvme_helper_fd, SNVME_HELP_GET_NVME_OFFSET, &mapping) < 0) {
        perror("ioctl failed");
        assert(0);
    }
    nvme_ofst_t nvme_ofst = mapping.address;
    close(snvme_helper_fd);
    close(fd);

    int *buf__dev;
    assert(cudaSuccess ==
            cudaMalloc(&buf__dev, block_size));

    read_from_nvme__using_device<<<1, 1>>>(nvme_ofst, buf__dev, block_size, 0);
    
    assert(cudaSuccess ==
            cudaMemcpy(buf__host, buf__dev, block_size, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < block_size / sizeof(int); i++)
        assert(buf__host[i] == i);

    for (size_t i = 0; i < block_size / sizeof(int); i++)
        buf__host[i] = i + 1;
    assert(cudaSuccess ==
            cudaMemcpy(buf__dev, buf__host, block_size, cudaMemcpyHostToDevice));
    write_to_nvme__using_device<<<1, 1>>>(nvme_ofst, buf__dev, block_size, 0);
    sync_nvme__using_device<<<1, 1>>>(nvme_ofst);

    // fd = open(filename, O_RDWR);
    // assert(0 <= fd);

    // assert(block_size == read(fd, buf__host, block_size));
    // for (size_t i = 0; i < block_size / sizeof(int); i++)
    //     std::cout << i << std::endl;
    // for (size_t i = 0; i < block_size / sizeof(int); i++)
    //     assert(buf__host[i] == i + 1);

    // close(fd);

    return 0;
}
