#!/bin/bash -x


TORCH_INCLUDE_DIR=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`/../../include
TORCH_CXX_INCLUDE="-I${TORCH_INCLUDE_DIR} -I${TORCH_INCLUDE_DIR}/torch/csrc/api/include"

/usr/local/cuda/bin/nvcc \
	-O2 \
	-I/home/hyf/Ganymede/Comparision/gpufs/include \
	-I/usr/local/cuda/include \
	-L/home/hyf/Ganymede/Comparision/gpufs/lib \
	-L/usr/local/cuda/lib64 \
	-lgpufs -lcudart \
	--std=c++17 \
	--generate-code code=sm_80,arch=compute_80 \
	-maxrregcount 32 \
	-Xcompiler=-fPIC \
	-dc api_gpu.cu -o api_gpu.o

g++ \
	-I/home/hyf/Ganymede/Comparision/interface_for_pytorch/include \
	-std=c++17 \
	$(python3 -m pybind11 --includes) \
	-I/home/hyf/Ganymede/Comparision/gpufs/include \
	-I/usr/local/cuda/include \
	${TORCH_CXX_INCLUDE} \
	-fPIC \
	-c \
	-o api.o \
	api.cpp

g++ \
	-shared \
	-fPIC \
	api_gpu.o api.o \
	-o api_gpu$(python3-config --extension-suffix)

stubgen -m api_gpu -o .

#./py_call_cpp.py
