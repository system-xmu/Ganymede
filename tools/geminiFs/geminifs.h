#ifndef GEMINIFS_H

#include <stdint.h>

typedef uint64_t vaddr_t;
typedef uint64_t rawfile_ofst_t;
typedef uint64_t nvme_ofst_t;

struct geminiFS_table_entry {
  rawfile_ofst_t raw_file_ofst;
  nvme_ofst_t nvme_ofst;
};

struct geminiFS_hdr {
  uint64_t magic_num;
  uint64_t virtual_space_size; /* in bytes */
  uint8_t block_bit; /* block_size, in the form of bit num */
  uint8_t table_entry_bit; /* number of table entry per block, in the form of bit num */
  uint32_t table_level_nr;
  struct geminiFS_table_entry l1_table;
  int fd; /* dummy */
};

extern union geminiFS_magic {
  uint64_t magic_num;
  char magic_cstr[8];
} the_geminiFS_magic;

typedef int fd_t;
typedef struct {void *rawfile_base__dev;rawfile_ofst_t l1_table_base;} device_fd_t;
typedef struct geminiFS_hdr *host_fd_t;

//-----------------host only------------------
extern host_fd_t
host_create_geminifs_file(const char *filename,
                          uint64_t block_size,
                          uint64_t virtual_space_size);

extern host_fd_t
host_open_geminifs_file(const char *filename);

extern size_t
host_xfer_geminifs_file(host_fd_t fd_1,
                        vaddr_t va,
                        void *buf_1,
                        size_t nbyte,
                        int is_read);

extern void
host_refine_nvmeofst(host_fd_t fd);

extern void
host_close_geminifs_file(host_fd_t fd);

//-----------------host for device------------------
extern device_fd_t
host_open_geminifs_file_for_device(host_fd_t);

#define GEMINIFS_H
#endif
