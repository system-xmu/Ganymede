NVCCFLAGS=-I../../src/include -I../../src/freestanding/include

example:
	nvcc $(NVCCFLAGS) example_for_nvme_raw_access.cu

a.out:  geminifs.cuda.o geminifs.o
	nvcc geminifs.cuda.o geminifs.o

geminifs.cuda.o: geminifs.h geminifs.cu
	nvcc geminifs.cu -c -o geminifs.cuda.o

geminifs.o: geminifs.h geminifs.c
	gcc -I/usr/local/cuda/include -lcudart -L/usr/local/cuda/targets/x86_64-linux/lib  geminifs.c -c -g -O0 -o geminifs.o

clean:
	rm -fr *.cuda.o *.o *.out