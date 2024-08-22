#/* 
#* This expermental software is provided AS IS. 
#* Feel free to use/modify/distribute, 
#* If used, please retain this disclaimer and cite 
#* "GPUfs: Integrating a file system with GPUs", 
#* M Silberstein,B Ford,I Keidar,E Witchel
#* ASPLOS13, March 2013, Houston,USA
#*/

CUDA_ROOT:=/usr/local/cuda
GPUFSROOTDIR?=../../Comparision/gpufs/

GPUFSINCLUDEDIR=$(GPUFSROOTDIR)/include
GPUFSLIBDIR=$(GPUFSROOTDIR)/lib

CUDA_INCLUDE=-I$(GPUFSROOTDIR)/include -I$(CUDA_ROOT)/include
CUDA_LIB=-L$(GPUFSROOTDIR)/lib -L$(CUDA_ROOT)/lib64 -lgpufs -lcudart

CXXFLAGS=-O2 $(CUDA_INCLUDE) $(CUDA_LIB)
NVCCFLAGS=-O2 $(CUDA_INCLUDE) $(CUDA_LIB) --std=c++11 --generate-code code=sm_80,arch=compute_80 -maxrregcount 32

NVCC=$(CUDA_ROOT)/bin/nvcc

test_read_latency: test_read_latency.o
	$(NVCC) $(NVCCFLAGS) --cudart static -link -o "test_read_latency" test_read_latency.o

test_read_latency.o: test_read_latency.cu
	$(NVCC) $(NVCCFLAGS) -dc test_read_latency.cu -o test_read_latency.o


PHONY:  clean

clean:
	rm -f test_read_latency *.o *.a
