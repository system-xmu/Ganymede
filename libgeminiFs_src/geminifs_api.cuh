#ifndef GEMINIFS_CUH
#define GEMINIFS_CUH

#include "geminifs_api.h"


extern __global__ void
device_xfer_geminifs_file(dev_fd_t fd_1,
                          vaddr_t va,
                          void *buf_dev,
                          size_t nbyte,
                          int is_read);


extern __global__ void
device_xfer_geminifs_file__batching_pagecache
(dev_fd_t *dev_fds,
 vaddr_t va,
 void *buf_dev,
 size_t nbyte,
 int is_read);

extern __global__ void
device_sync(dev_fd_t);

#endif
