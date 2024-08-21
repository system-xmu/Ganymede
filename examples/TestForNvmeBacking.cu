
#include <cassert>
#include <iostream>
#include <ctime>
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


#define geminifs_file_name "checkpoint.geminifs"
#define geminifs_file_path (nvme_mount_path "/" geminifs_file_name)


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

  size_t virtual_space_size = 1 * (1ull << 20)/*GB*/;
  size_t page_capacity = 16 * (1ull << 10);
  size_t file_block_size = 4096;
  size_t dev_page_size = 4096;

  srand(time(0));
  int rand_start = rand();

  remove(geminifs_file_path);

  host_fd_t host_fd = host_create_geminifs_file(geminifs_file_path, file_block_size, virtual_space_size);
  host_refine_nvmeofst(host_fd);

  uint64_t *buf1 = (uint64_t *)malloc(virtual_space_size);
  for (size_t i = 0; i < virtual_space_size / sizeof(uint64_t); i++)
      buf1[i] = rand_start + i;
  host_xfer_geminifs_file(host_fd, 0, buf1, virtual_space_size, 0);
  
  uint64_t *buf2 = (uint64_t *)malloc(virtual_space_size);
  host_xfer_geminifs_file(host_fd, 0, buf2, virtual_space_size, 1);
  //printf("buf2: ");
  //for (size_t i = 0; i < virtual_space_size / sizeof(uint64_t); i++)
  //    printf("%llx ", buf2[i]);
  //printf("\n");

  host_close_geminifs_file(host_fd);

  host_fd = host_open_geminifs_file(geminifs_file_path);

  uint64_t *dev_buf1;
  uint64_t *dev_buf2;

  //gpuErrchk(cudaMallocManaged(&dev_buf1, virtual_space_size));
  gpuErrchk(cudaMallocManaged(&dev_buf2, virtual_space_size));

  dev_fd_t dev_fd = host_open_geminifs_file_for_device(host_fd, page_capacity, dev_page_size);

  //device_xfer_geminifs_file<<<108, 32>>>(dev_fd, 0, dev_buf1, virtual_space_size, 0);
  //cudaDeviceSynchronize();
  device_xfer_geminifs_file<<<108, 32>>>(dev_fd, 0, dev_buf2, virtual_space_size, 1);
  cudaDeviceSynchronize();

  //uint64_t *buf3 = (uint64_t *)malloc(virtual_space_size);

  for (size_t i = 0; i < virtual_space_size / sizeof(uint64_t); i++) {
      if (dev_buf2[i] != i + rand_start) {
          printf("ERROR GPU!!!\n");
          //for (size_t j = 0; j < virtual_space_size / sizeof(uint64_t); j++) {
          //    printf("%llx ", (uint64_t)dev_buf2[j]);
          //}

          goto out;
      }
  }
  
  device_sync<<<1, 1>>>(dev_fd);
  cudaDeviceSynchronize();


  host_close_geminifs_file(host_fd);

  //host_fd = host_open_geminifs_file(geminifs_file_path);

  //host_xfer_geminifs_file(host_fd, 0, buf3, virtual_space_size, 1);


  //for (size_t i = 0; i < virtual_space_size / sizeof(uint64_t); i++) {
  //    printf("buf3[%llx] = %llx\n", (uint64_t)i, (uint64_t)buf3[i]);
  //    if (buf3[i] != i + 3) {
  //        printf("ERROR!!!");
  //        for (size_t j = 0; j < virtual_space_size / sizeof(uint64_t); j++) {
  //            printf("%llx ", (uint64_t)buf3[j]);
  //        }
  //        goto out;
  //    }
  //}
  printf("ALL OK!\n");

out:
  host_close_all();

  return 0;
}

