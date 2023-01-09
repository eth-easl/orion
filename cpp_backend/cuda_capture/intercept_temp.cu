#include "intercept_temp.h"

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
pthread_mutex_t mutex0;
int i=0;


cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {


	//printf("Captured a cudaLaunchKernel! id is %d, function ptr is %p, stream is %d, gridDim is %d, blockDim is %d, sharedMem is %ld\n", i, func, stream, gridDim, blockDim, sharedMem);


	cudaError_t (*function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaLaunchKernel");

	cudaError_t err = cudaSuccess;

	struct kernel_record new_record = {func, gridDim, blockDim, args, sharedMem, stream, false, 0};


	// push at queue
	pthread_mutex_lock(&mutex0);
	kqueue0.push(new_record);
	pthread_mutex_unlock(&mutex0);

	// wait and run
	while (true) {
		pthread_mutex_lock(&mutex0);
		if (kqueue0.front().run) {
			cudaStream_t sched_stream = kqueue0.front().sched_stream;
			kqueue0.pop();
			printf("-------- run with stream %d!!!\n", sched_stream);
			pthread_mutex_unlock(&mutex0);
			err = (*function)(func, gridDim, blockDim, args, sharedMem, sched_stream); 
			return err;
		}
		pthread_mutex_unlock(&mutex0);
	}
}

int main() {

	printf("running......\n");
}
