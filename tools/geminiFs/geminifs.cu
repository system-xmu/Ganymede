#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

extern "C" {
#include "geminifs.h"
}

__device__ static nvme_ofst_t
device__convert_va__to(int lv_nr,
                       int block_bit,
                       int table_entry_bit,
                       nvme_ofst_t l1_table_nvme_ofst,
                       vaddr_t va) {
  uint64_t block_size = 1 << block_bit;

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

  nvme_ofst_t cur_block_base = l1_table_nvme_ofst;
  while (cur_lv <= lv_nr) {
    int in_table_idx = table_idxes >> 64 - table_entry_bit;
    struct geminiFS_table_entry the_entry;
    // read in_table_idx
    //assert((off_t)(-1) !=
    //  lseek(fd, cur_block_base + in_table_idx * sizeof(the_entry), SEEK_SET)
    //);
    //assert(sizeof(the_entry) ==
    //  read(fd, &the_entry, sizeof(the_entry))
    //);

    if (the_entry.raw_file_ofst == 0) {
      assert(0);
    }

    cur_block_base = the_entry.raw_file_ofst;
    cur_lv++;
    table_idxes = table_idxes << table_entry_bit;
  }

  return cur_block_base + in_block_ofst;
}

__device__ static void
device_write_block(int lv_nr,
                   int block_bit,
                   int table_entry_bit,
                   nvme_ofst_t l1_table_nvme_ofst,
                   vaddr_t va,
                   char *buf) {
  nvme_ofst_t block_base = device__convert_va__to(lv_nr,
                                                  block_bit,
                                                  table_entry_bit,
                                                  l1_table_nvme_ofst,
                                                  va);
  uint64_t block_size = 1 << block_bit;
  //write block
}

__device__ static void
device_read_block(int lv_nr,
                  int block_bit,
                  int table_entry_bit,
                  nvme_ofst_t l1_table_nvme_ofst,
                  vaddr_t va,
                  char *buf) {
  nvme_ofst_t block_base = device__convert_va__to(lv_nr,
                                                  block_bit,
                                                  table_entry_bit,
                                                  l1_table_nvme_ofst,
                                                  va);
  uint64_t block_size = 1 << block_bit;
  //read block
}
