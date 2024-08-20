NVCCFLAGS=-arch=sm_80 -I../libgeminiFs_src -O3
LDFLAGS=-L../build/lib -Xlinker -rpath=$(shell pwd)/../build/lib -lgeminiFs -lnvm -arch=sm_80 -I../libgeminiFs_src

all: TestForNoBackingPagecache.exe TestForNvmeBacking.exe TestForPageCache.exe TestForNvmeofstRefining.exe

clean:
	rm -f *.exe

TestForNvmeofstRefining.exe: TestForNvmeofstRefining.cu
	nvcc $(LDFLAGS) $(NVCCFLAGS) TestForNvmeofstRefining.cu -o TestForNvmeofstRefining.exe

TestForNvmeBacking.exe: TestForNvmeBacking.cu
	nvcc $(LDFLAGS) $(NVCCFLAGS) TestForNvmeBacking.cu -o TestForNvmeBacking.exe

TestForNoBackingPagecache.exe: TestForNoBackingPagecache.cu
	nvcc $(LDFLAGS) $(NVCCFLAGS) TestForNoBackingPagecache.cu -o TestForNoBackingPagecache.exe

TestForPageCache.exe: TestForPageCache.cu
	nvcc $(LDFLAGS) $(NVCCFLAGS) TestForPageCache.cu -o TestForPageCache.exe
