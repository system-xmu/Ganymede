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
  rawfile_ofst_t l1_table_rawfile_ofst = dev_fd.l1_table_base;
  auto lv_nr = dev_fd.table_lv_nr;
  auto block_bit = dev_fd.block_bit;
  auto table_entry_bit = dev_fd.table_entry_bit;
  //uint64_t block_size = 1 << block_bit;

  // view of va:
  //       | entry_bit | entry_bit | ... | entry_bit |<- block_bit ->|
  // +-----+-----------+-----------+-----+-----------+---------------+
  // |     | idx in L1 | idx in L2 | ... | idx in Ln | in-block ofst |
  // +-----+-----------+-----------+-----+-----------+---------------+
  // |<----------------------------  64  --------------------------->|
  vaddr_t in_block_ofst = va & ((1 << block_bit) - 1);

  int cur_lv = 1;
  vaddr_t table_idxes =
    va << (64 - lv_nr * table_entry_bit - block_bit);
  // view of table_idxes:
  // |<-  entry_bit   ->|     |<-  entry_bit  ->|
  // +------------------+-----+-----------------+---------------+
  // | idx in L`cur_lv` | ... | idx in L`lv_nr` |               |
  // +------------------+-----+-----------------+---------------+
  // |<--------------------------  64  ------------------------>|

  rawfile_ofst_t cur_table_base = l1_table_rawfile_ofst;
  while (cur_lv <= lv_nr) {
    int in_table_idx = table_idxes >> (64 - table_entry_bit);
    struct geminiFS_table_entry *the_entry =
      (struct geminiFS_table_entry *)((uint64_t)(dev_fd.rawfile_base__dev) + cur_table_base) +
      in_table_idx;

    // read in_table_idx
    //assert((off_t)(-1) !=
    //  lseek(fd, cur_block_base + in_table_idx * sizeof(the_entry), SEEK_SET)
    //);
    //assert(sizeof(the_entry) ==
    //  read(fd, &the_entry, sizeof(the_entry))
    //);

    if (the_entry->raw_file_ofst == 0) {
      assert(0);
    }

    if (cur_lv == lv_nr) {
      return the_entry->nvme_ofst + in_block_ofst;
    }

    // here 'cur_lv < lv_nr'

    cur_table_base = the_entry->raw_file_ofst;
    cur_lv++;
    table_idxes = table_idxes << table_entry_bit;
  }

  assert(0); //unreachable
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


