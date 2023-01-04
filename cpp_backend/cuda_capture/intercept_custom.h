#include <dlfcn.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <queue>
#include <pthread.h>

struct kernel_record {

	const void* func;
	dim3 gridDim;
	dim3 blockDim;
	void** args;
	size_t sharedMem;
	cudaStream_t stream;
};


