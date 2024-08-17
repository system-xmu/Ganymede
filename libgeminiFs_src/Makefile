CCFLAGS=-O3 -I/usr/local/cuda/include -L/usr/local/cuda/targets/x86_64-linux/lib
NVCCFLAGS=-arch=sm_80 -IcuCollections/include -Icccl/thrust -Icccl/libcudacxx/include -Icccl/cub  -I../src/include  -I../src/freestanding/include --expt-relaxed-constexpr --expt-extended-lambda -rdc=true
LDFLAGS=-L. -L../build/lib -Xlinker -rpath=../build/lib -lgeminiFs -lnvm

all: libgeminiFs.a TestForNoBackingPagecache.exe TestForNvmeBacking.exe

libgeminiFs.a: geminifs.cuda.o geminifs_nvme.cuda.o geminifs.o
	ar rcs libgeminiFs.a geminifs.cuda.o geminifs_nvme.cuda.o geminifs.o

TestForNvmeBacking.exe: TestForNvmeBacking.cu libgeminiFs.a
	nvcc $(LDFLAGS) TestForNvmeBacking.cu -o TestForNvmeBacking.exe

TestForNoBackingPagecache.exe: TestForNoBackingPagecache.cu libgeminiFs.a
	nvcc $(LDFLAGS) TestForNoBackingPagecache.cu -o TestForNoBackingPagecache.exe

geminifs.cuda.o: geminifs_api.cuh geminifs_api.h geminifs.cu  geminifs_internal.cuh
	nvcc $(NVCCFLAGS) geminifs.cu  -c -o geminifs.cuda.o

geminifs_nvme.cuda.o: geminifs_api.cuh geminifs_api.h geminifs_nvme.cu geminifs_internal.cuh
	nvcc $(NVCCFLAGS) geminifs_nvme.cu -c -o geminifs_nvme.cuda.o

geminifs.o: geminifs_api.h geminifs.c
	gcc $(CCFLAGS) geminifs.c -c -o geminifs.o

clean:
	rm -f *.o  *.a *.exe
