
#include <cassert>
#include <iostream>
#include "geminifs_api.cuh"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define snvme_control_path "/dev/snvm_control"
#define snvme_path "/dev/snvme2"
#define nvme_mount_path "/home/qs/nvm_mount"
#define nvme_dev_path "/dev/nvme1n1"

static void
test_host(host_fd_t host_fd) {
  size_t size = host_fd->virtual_space_size;
  size_t *buf1 = (size_t *)malloc(size);
  size_t *buf2 = (size_t *)malloc(size);

  for (size_t i = 0; i < size/sizeof(size_t); i++) {
    buf1[i] = i + 2;
  }

  host_xfer_geminifs_file(host_fd, 0, buf1, size, 0);
  host_xfer_geminifs_file(host_fd, 0, buf2, size, 1);

  for (size_t i = 0; i < size/sizeof(size_t); i++) {
    assert(buf1[i] == buf2[i]);
  }

  free(buf1);
  free(buf2);
}


int
main() {
    host_open_all(
            snvme_control_path,
            snvme_path,
            nvme_dev_path,
            nvme_mount_path,
            1,
            1024,
            32);

  size_t virtual_space_size = 20 * (1ull << 30)/*GB*/;
  size_t page_capacity = 128 * (1ull << 20);
  size_t file_block_size = 4096;
  size_t dev_page_size = 4096;
  //size_t virtual_space_size = 4096;
  host_fd_t host_fd = host_create_geminifs_file(nvme_mount_path "/checkpoint.geminifs", file_block_size, virtual_space_size);

  test_host(host_fd);

  host_close_geminifs_file(host_fd);

  host_fd = host_open_geminifs_file("checkpoint.geminifs");
  test_host(host_fd);

  host_refine_nvmeofst(host_fd);
  host_close_geminifs_file(host_fd);


  host_fd = host_open_geminifs_file("checkpoint.geminifs");

  uint64_t *dev_buf1;
  uint64_t *dev_buf2;

  gpuErrchk(cudaMallocManaged(&dev_buf1, virtual_space_size));
  gpuErrchk(cudaMallocManaged(&dev_buf2, virtual_space_size));

  for (size_t i = 0; i < virtual_space_size / sizeof(uint64_t); i++)
      dev_buf1[i] = i + 3;

  dev_fd_t dev_fd = host_open_geminifs_file_for_device(host_fd, page_capacity, dev_page_size);

  device_xfer_geminifs_file<<<108, 32>>>(dev_fd, 0, dev_buf1, virtual_space_size, 0);
  device_xfer_geminifs_file<<<108, 32>>>(dev_fd, 0, dev_buf2, virtual_space_size, 1);

  for (size_t i = 0; i < virtual_space_size / sizeof(uint64_t); i++)
      dev_buf2[i] = i + 3;
  
  device_sync<<<1, 1>>>(dev_fd);

  uint64_t *buf3 = (uint64_t *)malloc(virtual_space_size);
  host_xfer_geminifs_file(host_fd, 0, buf3, virtual_space_size, 1);
  for (size_t i = 0; i < virtual_space_size / sizeof(uint64_t); i++)
      buf3[i] = i + 3;


  host_close_geminifs_file(host_fd);


  host_close_all();

  return 0;
}

