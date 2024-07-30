#include "api_gpu.cuh"

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int) err, cudaGetErrorString(err));
		exit(-1);
	}
}

__global__ void read(char* filename, size_t offset, size_t size, uchar* buffer)
{
    __shared__ int fd;
    fd = 0;
    fd = gopen(filename, O_GRDONLY);
    __syncthreads(); // Ensure all threads have the same fd value
    if (fd < 0)
        ERROR("Failed to open file");
    size_t bytes_read=gread(fd,offset, size, buffer);
    if (bytes_read!=size) 
        ERROR("Failed to read data");
    if(gclose(fd) < 0)
        ERROR("Failed to close file");
}

__global__ void write(char* filename, size_t offset, size_t size, uchar *data)
{
    __shared__ int fd;
    fd = 0;
    fd = gopen(filename, O_GRDONLY);
    __syncthreads(); // Ensure all threads have the same fd value
    if (fd < 0)
        ERROR("Failed to open file");
    size_t bytes_write = gwrite(fd, offset, size, data);
    if(bytes_write != size)
        ERROR("Failed to write data");
    if(gclose(fd) < 0)
        ERROR("Failed to close file");
}

void gpu_write(char* filename, size_t offset, size_t size, uchar *data)
{
    write<<<1,1>>>(filename, offset, size, data);
    return;
}
void gpu_read(char* filename, size_t offset, size_t size, uchar *buffer)
{
    read<<<1,1>>>(filename, offset, size, buffer);
    return;
}


int main()
{
    return 0;
}