#include "intercept_custom.h"

#define FUNC_DEF(func) { #func }


using namespace std;

void print_kernel_invocation(int i, dim3 gridDim, dim3 blockDim) {
	
	printf("%d, ", i);
	if (gridDim.y == 1 && gridDim.z == 1) {
  		printf("--gridDim=%d ", gridDim.x);
	} else if (gridDim.z == 1) {
		printf("--gridDim=[%d,%d] ", gridDim.x, gridDim.y);
	} else {
		printf("--gridDim=[%d,%d,%d] ", gridDim.x, gridDim.y, gridDim.z);
	}

	if (blockDim.y == 1 && blockDim.z == 1) {
		printf("--blockDim=%d ", blockDim.x);
	} else if (blockDim.z == 1) {
		printf("--blockDim=[%d,%d] ", blockDim.x, blockDim.y);
	} else {
		printf("--blockDim=[%d,%d,%d] ", blockDim.x, blockDim.y, blockDim.z);
	}
	//printf("\n");
}


cudaError_t cudaMalloc(void** devPtr, size_t size) {

	printf("Caught cudaMalloc!\n");

	cudaError_t (*function)(void** devPtr, size_t size);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMalloc");
	
	cudaError_t err = (*function)(devPtr, size);
	return err;

}

queue<kernel_record> kqueue0;
queue<kernel_record> kqueue1;
pthread_t thread_ids[2];
pthread_mutex_t mutex0;
pthread_mutex_t mutex1;
int i=0;

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
	printf("Captured a cuGetProcAddress\n");
}	


CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra ) {

	printf("Captured a cuLaunchKernel\n");

}

cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {


	//printf("Captured a cudaLaunchKernel! id is %d, function ptr is %p, stream is %d, gridDim is %d, blockDim is %d, sharedMem is %ld, args is %p\n", i, func, stream, gridDim, blockDim, sharedMem, args);


	cudaError_t (*function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaLaunchKernel");

	//printf(", %p\n", function);

	cudaError_t err = cudaSuccess;
	if (stream != 0) {
		//printf("------------- RUN!!!!!!!!!!!!!, args is %p\n", args);
		
		printf("%p, ", func);
		print_kernel_invocation(i, gridDim, blockDim);
		printf("\n");

		err = (*function)(func, gridDim, blockDim, args, sharedMem, stream);
		printf("------------- done!\n");
		i+=1;
		return err;
	}

	//printf("args is %p\n", args);
	struct kernel_record new_record = {func, gridDim, blockDim, args, sharedMem, stream};
	

	if (1) { //mytid == thread_ids[0]) {
		pthread_mutex_lock(&mutex0);
		kqueue0.push(new_record);
		pthread_mutex_unlock(&mutex0);
		//printf("Kqueue0 - address: %p, size: %d, %d, %d, %d\n", &kqueue0, kqueue0.size(), thread_ids[0], thread_ids[1], stream);
	}
	else {
		pthread_mutex_lock(&mutex1);
		kqueue1.push(new_record);
		pthread_mutex_unlock(&mutex1);
		printf("Kqueue1 - address: %p, size: %d, %d, %d, %d\n", &kqueue1, kqueue1.size(), thread_ids[0], thread_ids[1], stream);
	}

}

int main() {

	printf("running......\n");
}
