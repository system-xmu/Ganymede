NVCCFLAGS=-arch=sm_80 -I./cuCollections/include -Icccl/thrust -Icccl/libcudacxx/include -Icccl/cub --expt-relaxed-constexpr --expt-extended-lambda -rdc=true -diag-suppress=68  -g -G

#example:
#	nvcc $(NVCCFLAGS) example_for_nvme_raw_access.cu

a.out:  geminifs.cuda.o geminifs.o
	nvcc $(NVCCFLAGS) geminifs.cuda.o geminifs.o

geminifs.cuda.o: geminifs_api.cuh geminifs_api.h geminifs.cu geminifs_internal.cuh
	nvcc $(NVCCFLAGS) geminifs.cu -c -o geminifs.cuda.o

geminifs.o: geminifs_api.h geminifs.c
	gcc -I/usr/local/cuda/include -lcudart -L/usr/local/cuda/targets/x86_64-linux/lib  geminifs.c -c -g -O0 -o geminifs.o

clean:
	rm *.o *.out
