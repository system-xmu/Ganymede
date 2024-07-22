#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

extern "C" {
#include "geminifs.h"
}

__device__ static nvme_ofst_t
device__convert_va__to(dev_fd_t dev_fd,
                       vaddr_t va) {
  auto nr_l1 = dev_fd.nr_l1;
  auto block_bit = dev_fd.block_bit;

  uint64_t l1_idx = va >> block_bit;

  if (l1_idx < nr_l1)
    return dev_fd.l1__dev[l1_idx];

  return 0;
}

__global__ void
device_convert_va(dev_fd_t dev_fd,
                  vaddr_t va,
                  nvme_ofst_t *ret__dev) {
  *ret__dev = device__convert_va__to(dev_fd, va);
}
  
nvme_ofst_t
host_convert_va__using_device(dev_fd_t dev_fd,
        vaddr_t va) {
  nvme_ofst_t *ret__dev;
  assert(cudaSuccess ==
    cudaMalloc(&ret__dev, sizeof(nvme_ofst_t))
  );

  device_convert_va<<<1, 1>>>(dev_fd, va, ret__dev);

  nvme_ofst_t ret;
  assert(cudaSuccess ==
    cudaMemcpy(
      &ret,
      ret__dev,
      sizeof(nvme_ofst_t),
      cudaMemcpyDeviceToHost)
  );

  assert(cudaSuccess == cudaFree(ret__dev));
  return ret;
}


//__device__ static void
//device_write_block(int lv_nr,
//                   int block_bit,
//                   int table_entry_bit,
//                   nvme_ofst_t l1_table_nvme_ofst,
//                   vaddr_t va,
//                   char *buf) {
//  nvme_ofst_t block_base = device__convert_va__to(lv_nr,
//                                                  block_bit,
//                                                  table_entry_bit,
//                                                  l1_table_nvme_ofst,
//                                                  va);
//  uint64_t block_size = 1 << block_bit;
//  //write block
//}
//
//__device__ static void
//device_read_block(int lv_nr,
//                  int block_bit,
//                  int table_entry_bit,
//                  nvme_ofst_t l1_table_nvme_ofst,
//                  vaddr_t va,
//                  char *buf) {
//  nvme_ofst_t block_base = device__convert_va__to(lv_nr,
//                                                  block_bit,
//                                                  table_entry_bit,
//                                                  l1_table_nvme_ofst,
//                                                  va);
//  uint64_t block_size = 1 << block_bit;
//  //read block
//}


