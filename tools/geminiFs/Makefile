a.out: geminifs.h geminifs.c geminifs.cu
#nvcc geminifs.cu -c
	gcc -I/usr/local/cuda/include -lcudart -L/usr/local/cuda/targets/x86_64-linux/lib  geminifs.c -g -O0
