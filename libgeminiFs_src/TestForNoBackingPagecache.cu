
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


int
main() {
  int block_size = 1ull << 12; /* 4096 */
  size_t capacity = 128 * (1ull << 20); /* 128M */

  gpuErrchk(cudaSetDevice(0));

  size_t heapsz = 1 * (1ull << 30);
  gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapsz));

  dev_fd_t dev_fd =
      host_open_geminifs_file_for_device_without_backing_file(block_size, capacity);

  size_t buf_size = 128 * (1ull << 10); /* 128k */

  uint64_t *whole_host_buf = (uint64_t *)malloc(capacity);
  uint64_t *another_whole_host_buf = (uint64_t *)malloc(capacity);

  uint64_t *dev_buf1;
  uint64_t *dev_buf2;
  gpuErrchk(cudaMalloc(&dev_buf1, buf_size));
  gpuErrchk(cudaMalloc(&dev_buf2, buf_size));

  for (size_t i = 0; i < capacity / sizeof(uint64_t); i++)
    whole_host_buf[i] = i + 2;

  for (vaddr_t va = 0; va < capacity; va += buf_size) {
      gpuErrchk(cudaMemcpy(dev_buf1, (uint8_t *)whole_host_buf + va, buf_size, cudaMemcpyHostToDevice));
      cudaDeviceSynchronize();
      device_xfer_geminifs_file<<<54, 32>>>(dev_fd, va, dev_buf1, buf_size, 0);
      cudaDeviceSynchronize();
      exit(0);
  }

  for (vaddr_t va = 0; va < capacity; va += buf_size) {
      device_xfer_geminifs_file<<<54, 32>>>(dev_fd, va, dev_buf2, buf_size, 1);
      cudaDeviceSynchronize();
      gpuErrchk(cudaMemcpy((uint8_t *)another_whole_host_buf + va, dev_buf2, buf_size, cudaMemcpyDeviceToHost));
      cudaDeviceSynchronize();
  }

  for (size_t i = 0; i < capacity / sizeof(uint64_t); i++) {
      assert(another_whole_host_buf[i] == i + 2);
  }

  return 0;
}

