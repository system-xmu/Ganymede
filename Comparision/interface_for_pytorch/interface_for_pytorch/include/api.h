
#include <stdio.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <unistd.h>
#include <fcntl.h>
#include <string>
#include <stdexcept>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <map>
#include <cstring> 
// #include "api_gpu.cuh"
#include "space_mgr.h"

typedef unsigned char uchar;  

extern "C" void gpu_write(char* filename, size_t offset, size_t size, uchar *data);
extern "C" void gpu_read(char* filename, size_t offset, size_t size, uchar *buffer);

class GPUfs
{
public:
    GPUfs(const std::string &filename, const std::string &backend = "GPUfs");

    void gpufs_write(const at::Tensor &tensor, const std::string &key);
    void gpufs_read(const at::Tensor &tensor, const std::string &key);

    SpaceInfo prepare_write(const at::Tensor &tensor, const std::string &key);
    SpaceInfo prepare_read(const at::Tensor &tensor, const std::string &key);
    ~GPUfs();

private:
    const std::string filename;
    const std::string backend;
    // 管理tensor分配的信息
    SpaceManager space_mgr;
    std::unordered_map<std::string, SpaceInfo> tensors_info;
    void release(ull offset, ull bytes);

};

