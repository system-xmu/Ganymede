
#include "api.h"


// write需要写cuda Tensor
SpaceInfo GPUfs::prepare_write(const at::Tensor &tensor, const std::string &key)
{
    if (!tensor.is_contiguous() || !tensor.is_cuda())
        throw std::runtime_error("Tensor must be contiguous and on cuda");
    ull bytes = tensor.storage().nbytes();
    ull offset = this->space_mgr.alloc(bytes);
    SpaceInfo space_info(offset, bytes);
    this->tensors_info[key] = space_info;
    return space_info;
}

SpaceInfo GPUfs::prepare_read(const at::Tensor &tensor, const std::string &key)
{
    // TODO, 同时支持cuda 和cpu tensor
    if (!tensor.is_contiguous() || !tensor.is_cpu())
        throw std::runtime_error("Tensor must be contiguous and on cpu");
    if (this->tensors_info.find(key) == this->tensors_info.end())
        throw std::runtime_error("Read error, tensor not found");
    ull bytes = tensor.storage().nbytes();
    SpaceInfo space_info = this->tensors_info[key];
    if (bytes != space_info.second)
        throw std::runtime_error("Read error, tensor shape mismatch");
    this->tensors_info.erase(key);
    return space_info;
}

GPUfs::GPUfs(const std::string &filename, const std::string &backend): filename(filename), space_mgr(SpaceManager(0))
{
    // this->filename = filename;
    // this->backend = backend;
}
GPUfs::~GPUfs()
{
}
void GPUfs::gpufs_write(const at::Tensor &tensor, const std::string &key)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_write(tensor, key);
    printf("[C++]Tensor ptr:%p, offset: %lld, bytes: %lld\n", (void*)tensor.data_ptr(), offset, bytes);
    gpu_write(const_cast<char*>(this->filename.c_str()), offset, bytes, reinterpret_cast<char*>(tensor.data_ptr()));

    // lseek(this->fd, offset, SEEK_SET);
    // write(this->fd, tensor.data_ptr(), bytes);

}

void GPUfs::gpufs_read(const at::Tensor &tensor, const std::string &key)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_read(tensor, key);
    // lseek(this->fd, offset, SEEK_SET);
    // read(this->fd, tensor.data_ptr(), bytes);
    gpu_read(const_cast<char*>(this->filename.c_str()), offset, bytes, reinterpret_cast<char*>(tensor.data_ptr()));
    release(offset, bytes);
}

void GPUfs::release(ull offset, ull bytes)
{
    this->space_mgr.free(offset, bytes);
}
namespace py = pybind11;

PYBIND11_MODULE(_C, m)
{

    m.def("add", [](int a, int b) -> int {return a + b;});
    py::class_<GPUfs>(m, "GPUfs")
        .def(py::init<const std::string &, const std::string &>(), py::arg("filename"),  py::arg("backend") = "GPUfs")
        .def("gpufs_write", &GPUfs::gpufs_write, py::arg("tensor"), py::arg("key"))
        .def("gpufs_read", &GPUfs::gpufs_read, py::arg("tensor"), py::arg("key"));

}