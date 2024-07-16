#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>

#include <cuda_runtime.h>

#include "geminifs.h"
#include "../get-offset/get-offset.h"

union geminiFS_magic
the_geminiFS_magic = {
  .magic_cstr = {'g', 'e', 'm', 'i', 'n', 'i', 'f', 's'}
};

static int
one_nr__of__binary_int(unsigned long long i) {
  int count = 0;
  while (i != 0) {
    if ((i & 1) == 1)
      count++;
    i = i >> 1;
  }
  return count;
}

static void
init_geminiFS_hdr(struct geminiFS_hdr *hdr,
                  uint64_t block_size,
                  uint64_t virtual_space_size) {
  hdr->magic_num = the_geminiFS_magic.magic_num;
  hdr->virtual_space_size = virtual_space_size;
  hdr->block_bit = one_nr__of__binary_int(block_size - 1);

  assert(virtual_space_size % block_size == 0);

  int cur_table_level_nr = 0;
  uint64_t total_table_entry_nr__of__cur_lv = 1;
  int nr_table_entry__per_block = block_size / sizeof(struct geminiFS_table_entry);
  uint64_t max_geminiFS_virtual_space_size = total_table_entry_nr__of__cur_lv * block_size;
  while (max_geminiFS_virtual_space_size < virtual_space_size) {
    cur_table_level_nr++;
    total_table_entry_nr__of__cur_lv = total_table_entry_nr__of__cur_lv * nr_table_entry__per_block;
    max_geminiFS_virtual_space_size = total_table_entry_nr__of__cur_lv * block_size;
  }

  hdr->table_entry_bit = one_nr__of__binary_int(nr_table_entry__per_block - 1);

  if (cur_table_level_nr == 0)
    cur_table_level_nr++;
  hdr->table_level_nr = cur_table_level_nr;

  hdr->l1_table.raw_file_ofst = 0;
}

static rawfile_ofst_t
alloc_block(fd_t fd,
            uint64_t block_size) {
  off_t ret = lseek(fd, 0, SEEK_END);
  assert((off_t)(-1) != ret);
  assert(0 == ftruncate(fd, ret + block_size));

  return ret;
}

static rawfile_ofst_t
host__convert_va__to_1(int lv_nr,
                       int block_bit,
                       int table_entry_bit,
                       rawfile_ofst_t l1_table_raw_file_ofst,
                       fd_t fd,
                       vaddr_t va,
                       int alloc) {
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

  rawfile_ofst_t cur_block_base = l1_table_raw_file_ofst;
  while (cur_lv <= lv_nr) {
    int in_table_idx = table_idxes >> 64 - table_entry_bit;
    struct geminiFS_table_entry the_entry;
    assert((off_t)(-1) !=
      lseek(fd, cur_block_base + in_table_idx * sizeof(the_entry), SEEK_SET)
    );
    assert(sizeof(the_entry) ==
      read(fd, &the_entry, sizeof(the_entry))
    );

    if (the_entry.raw_file_ofst == 0) {
      assert(alloc);
      the_entry.raw_file_ofst = alloc_block(fd, block_size);
      assert((off_t)(-1) !=
        lseek(fd, cur_block_base + in_table_idx * sizeof(the_entry), SEEK_SET)
      );
      assert(sizeof(the_entry) ==
        write(fd, &the_entry, sizeof(the_entry))
      );
    }

    cur_block_base = the_entry.raw_file_ofst;
    cur_lv++;
    table_idxes = table_idxes << table_entry_bit;
  }

  return cur_block_base + in_block_ofst;
}

static rawfile_ofst_t
host__convert_va__to(struct geminiFS_hdr *hdr,
                     fd_t fd,
                     vaddr_t va,
                     int alloc) {
  int lv_nr = hdr->table_level_nr;
  int block_bit = hdr->block_bit;
  int table_entry_bit = hdr->table_entry_bit;
  rawfile_ofst_t l1_table_raw_file_ofst = hdr->l1_table.raw_file_ofst;
  return host__convert_va__to_1(lv_nr,
                                   block_bit,
                                   table_entry_bit,
                                   l1_table_raw_file_ofst,
                                   fd,
                                   va,
                                   alloc);
}

static void
host__for_each_table_entry_1(host_fd_t fd,
                             int cur_tb_lv,
                             rawfile_ofst_t raw_file_ofst,
                             void fun(struct geminiFS_table_entry *entry,
                                      int cur_tb_lv,
                                      void *context),
                             void *context) {
  size_t nr_table_entry = 1 << fd->table_entry_bit;
  size_t block_size = 1 << fd->block_bit;
  size_t table_level_nr = fd->table_level_nr;
  if (table_level_nr < cur_tb_lv)
    return;

  struct geminiFS_table_entry *table = mmap(NULL,
                  block_size,
                  PROT_WRITE | PROT_READ,
                  MAP_SHARED,
                  fd->fd,
                  raw_file_ofst);
  assert((void *) -1 != table);
  for (size_t i = 0; i < nr_table_entry; i++) {
    //printf("%d, ", cur_tb_lv);
    rawfile_ofst_t table_base = table[i].raw_file_ofst;
    if (table_base != 0) {
      fun(&(table[i]), cur_tb_lv, context);
      host__for_each_table_entry_1(fd, cur_tb_lv + 1, table_base, fun, context);
    }
  }

  munmap(table, block_size);
}

static void
host_for_each_table_entry(host_fd_t fd,
                           void fun(struct geminiFS_table_entry *entry,
                                    int cur_tb_lv,
                                    void *context),
                           void *context) {
  fun(&(fd->l1_table), 0, context);

  host__for_each_table_entry_1(fd, 1, fd->l1_table.raw_file_ofst, fun, context);

}

host_fd_t
host_create_geminifs_file(const char *filename,
                          uint64_t block_size,
                          uint64_t virtual_space_size) {
  struct geminiFS_hdr *hdr = malloc(sizeof(struct geminiFS_hdr));
  fd_t fd;

  fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  hdr->fd = fd;
  assert(0 <= fd);

  init_geminiFS_hdr(hdr, block_size, virtual_space_size);

  assert(0 < hdr->table_level_nr);
  rawfile_ofst_t hdr_ofst = alloc_block(fd, block_size);
  rawfile_ofst_t l1_ofst = alloc_block(fd, block_size);
  hdr->l1_table.raw_file_ofst = l1_ofst;
  assert((off_t)(-1) != lseek(fd, hdr_ofst, SEEK_SET));
  assert(sizeof(*hdr) == write(fd, hdr, sizeof(*hdr)));

  for (vaddr_t va = 0; va < virtual_space_size; va += block_size)
    host__convert_va__to(hdr, fd, va, 1);
  //for (vaddr_t va = 0; va < virtual_space_size; va += block_size) {
  //  rawfile_ofst_t raw_ofst = host__convert_va__to(&hdr, fd, va, 0);
  //  printf("va = %lx, raw_ofst = %lx\n", va, raw_ofst);
  //}

  return hdr;
}

host_fd_t
host_open_geminifs_file(const char *filename) {
  struct geminiFS_hdr *hdr = malloc(sizeof(struct geminiFS_hdr));
  fd_t fd;

  fd = open(filename, O_RDWR);
  assert(0 <= fd);

  assert((off_t)(-1) != lseek(fd, 0, SEEK_SET));
  assert(sizeof(*hdr) == read(fd, hdr, sizeof(*hdr)));

  hdr->fd = fd;


  assert(hdr->magic_num == the_geminiFS_magic.magic_num);

  return hdr;
}

static size_t
host_xfer_geminifs_file__in_one_page(int lv_nr,
                int block_bit,
                int table_entry_bit,
                rawfile_ofst_t l1_table_raw_file_ofst,
                fd_t fd, vaddr_t va, char *b, size_t nbyte, int block_size,
                int is_read) {
#define BLOCK_VA(va, block_size) (((size_t)(va) & (~((block_size) - 1))))
  vaddr_t va_end1 = BLOCK_VA(va + block_size, block_size);
  vaddr_t va_end2 = va + nbyte;
  vaddr_t va_end;
  if (va_end1 < va_end2)
    va_end = va_end1;
  else
    va_end = va_end2;

  nbyte = va_end - va;

  rawfile_ofst_t a = host__convert_va__to_1(lv_nr,
                                            block_bit,
                                            table_entry_bit,
                                            l1_table_raw_file_ofst,
                                            fd, va, 0);
  assert((off_t)(-1) != lseek(fd, a, SEEK_SET));
  if (is_read)
    assert(nbyte == read(fd, b, nbyte));
  else
    assert(nbyte == write(fd, b, nbyte));
  return nbyte;
}

size_t
host_xfer_geminifs_file(host_fd_t fd_1,
                        vaddr_t va,
                        void *buf_1,
                        size_t nbyte,
                        int is_read) {
  struct geminiFS_hdr *hdr = fd_1;
  assert(va + nbyte <= hdr->virtual_space_size);
  int fd = hdr->fd;
  int lv_nr = hdr->table_level_nr;
  int block_bit = hdr->block_bit;
  int table_entry_bit = hdr->table_entry_bit;
  rawfile_ofst_t l1_table_raw_file_ofst = hdr->l1_table.raw_file_ofst;
  char *buf = buf_1;

  int block_size = 1 << block_bit;

  size_t ret = 0;
  while (0 < nbyte) {
          size_t read_n = host_xfer_geminifs_file__in_one_page(lv_nr,
                          block_bit,
                          table_entry_bit,
                          l1_table_raw_file_ofst,
                          fd, va, buf, nbyte, block_size, is_read);
          ret += read_n;
          buf += read_n;
          va += read_n;
          nbyte -= read_n;
  }

  return ret;
}

void
host_close_geminifs_file(host_fd_t fd) {
  close(fd->fd);
  free(fd);
}

struct refine_nvmeofst_context {
    int fd_file;
    int snvme_helper_fd;
    int block_size;
};
static void
host_refine_nvmeofst_1(struct geminiFS_table_entry *e,
                       int cur_tb_lv,
                       void *context_1) {
  struct refine_nvmeofst_context *context = context_1;
  struct nds_mapping mapping;
  mapping.file_fd = context->fd_file;
  mapping.offset = e->raw_file_ofst;
  mapping.len = context->block_size;
  if (ioctl(context->snvme_helper_fd, SNVME_HELP_GET_NVME_OFFSET, &mapping) < 0) {
      perror("ioctl failed");
      assert(0);
  } 
  e->nvme_ofst = mapping.address;
  printf("%lx, %lx\n", e->raw_file_ofst, e->nvme_ofst);
}

#define snvme_helper_path "/dev/snvme_helper"
void
host_refine_nvmeofst(host_fd_t fd) {
  int fd_dev;
  fd_dev = open(snvme_helper_path, O_RDWR);
  if (fd_dev < 0) {
      perror("Failed to open fd_dev");
      assert(0);
  }
  struct refine_nvmeofst_context c;
  c.fd_file = fd->fd;
  c.snvme_helper_fd = fd_dev;
  c.block_size = 1 << fd->block_bit;
  host_for_each_table_entry(fd, host_refine_nvmeofst_1, &c);
  close(fd_dev);
}

static size_t
raw_file_size(int fd) {
  return lseek(fd, 0, SEEK_END);
}

struct host_open_geminifs_file_for_device__context {
  void *raw_file_base__on_device;
  rawfile_ofst_t l1_table_base;
  size_t block_size;
  int table_lv_nr;
  int fd;
};

static void
host_open_geminifs_file_for_device_1(struct geminiFS_table_entry *entry,
                                     int cur_tb_lv,
                                     void *context_1) {
  struct host_open_geminifs_file_for_device__context *c = context_1;
  if (cur_tb_lv == c->table_lv_nr)
    return;
  if (entry->raw_file_ofst == 0)
    return;

  if (cur_tb_lv == 0)
    c->l1_table_base = entry->raw_file_ofst;
  
  void *t_block = malloc(c->block_size);

  assert((off_t)(-1) !=
    lseek(c->fd, entry->raw_file_ofst, SEEK_SET)
  );
  assert(c->block_size ==
    read(c->fd, t_block, c->block_size)
  );
  assert(cudaSuccess ==
    cudaMemcpy(
      (void *)((uint64_t)(c->raw_file_base__on_device) + entry->raw_file_ofst),
      t_block,
      c->block_size,
      cudaMemcpyHostToDevice)
  );

  free(t_block);
}

device_fd_t
host_open_geminifs_file_for_device(host_fd_t host_fd) {
  device_fd_t ret;
  struct geminiFS_hdr *hdr = host_fd;
  void *raw_file_base__on_device;
  assert(cudaSuccess ==
    cudaMalloc(&raw_file_base__on_device, raw_file_size(hdr->fd)));

  struct host_open_geminifs_file_for_device__context t_context =
    {.raw_file_base__on_device = raw_file_base__on_device,
     .block_size = 1 << hdr->block_bit,
     .table_lv_nr = hdr->table_level_nr,
     .fd = hdr->fd};
  
  host_for_each_table_entry(host_fd,
          host_open_geminifs_file_for_device_1,
          &t_context);

  ret.rawfile_base__dev = raw_file_base__on_device;
  ret.l1_table_base = t_context.l1_table_base;
  return ret;
}

int
main() {
  //size_t size = (uint64_t)1 * (1 << 30)/*GB*/;
  size_t size = 4096;
  host_fd_t fd = host_create_geminifs_file("checkpoint.geminifs", 4096, size);

  size_t * buf1 = malloc(size);
  size_t * buf2 = malloc(size);

  for (size_t i = 0; i < size/sizeof(size_t); i++) {
    buf1[i] = i;
  }

  host_xfer_geminifs_file(fd, 0, buf1, size, 0);
  host_xfer_geminifs_file(fd, 0, buf2, size, 1);

  for (size_t i = 0; i < size/sizeof(size_t); i++) {
    assert(buf1[i] == buf2[i]);
  }
  host_xfer_geminifs_file(fd, 8, buf2, 4088, 1);
  for (size_t i = 0; i < 512; i++) {
  printf("%ld\n", buf2[i]);
  }
  host_close_geminifs_file(fd);

  fd = host_open_geminifs_file("checkpoint.geminifs");
  host_xfer_geminifs_file(fd, 0, buf2, size, 1);

  for (size_t i = 0; i < size/sizeof(size_t); i++) {
    assert(buf1[i] == buf2[i]);
  }
  host_refine_nvmeofst(fd);

  host_open_geminifs_file_for_device(fd);

  host_close_geminifs_file(fd);
  return 0;
}
