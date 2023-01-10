#include <dlfcn.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <queue>
#include <pthread.h>
#include <cuda.h>
#include <sys/types.h>
#include <syscall.h>
#include <unistd.h>
#include <sys/syscall.h>

struct kernel_record {

	const void* func;
	dim3 gridDim;
	dim3 blockDim;
	void** args;
	size_t sharedMem;
	cudaStream_t stream;
	volatile bool run;
	volatile cudaStream_t sched_stream;
};


