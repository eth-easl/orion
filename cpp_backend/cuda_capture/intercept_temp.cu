#include "intercept_temp.h"

using namespace std;

queue<kernel_record> kqueue0;
queue<kernel_record> kqueue1;
pthread_mutex_t mutex0;
pthread_mutex_t mutex1;
pthread_t thread_ids[2];

queue<kernel_record>* kqueues[2] = {&kqueue0, &kqueue1};
pthread_mutex_t* mutexes[2] = {&mutex0, &mutex1};
int i=0;

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

	printf("Caught cudaMalloc! allocate region of %ld bytes\n", *devPtr, size);

	cudaError_t (*function)(void** devPtr, size_t size);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMalloc");
	
	cudaError_t err = (*function)(devPtr, size);
	printf("Memory allocated at address %p\n", *devPtr);
	return err;

}


cudaError_t cudaFree(void* devPtr) {

	printf("Caught cudaFree! Free pointer that holds address %p\n", devPtr);

	cudaError_t (*function)(void* devPtr);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaFree");

	cudaError_t err; //= (*function)(devPtr);
	return err;

}

cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {


	printf("Captured a cudaLaunchKernel! id is %d, function ptr is %p, stream is %d, gridDim is %d, blockDim is %d, sharedMem is %ld\n", i, func, stream, gridDim, blockDim, sharedMem);


	cudaError_t (*function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaLaunchKernel");
	cudaError_t err = cudaSuccess;

	struct kernel_record new_record = {func, gridDim, blockDim, args, sharedMem, stream, false, 0};

	// push at queue

#ifdef SYS_gettid
	pid_t tid = syscall(SYS_gettid);
#else
#error "SYS_gettid unavailable on this system"
#endif
	printf("My id is %d\n", tid);

	int idx=-1;
	if (tid == thread_ids[0])
		idx = 0;
	else if (tid == thread_ids[1])
		idx = 1;
	else
		idx = 0;
		//printf("----------------------- INVALID!!!!!!!!! -------------------\n");

	//printf("idx: %d, queues: %p, queue: %p, mutex: %p\n", idx, kqueues, kqueues[idx], mutexes[idx]);

	if (stream==0) {
	
		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);
	}
	else {
		printf("------------------------ before submitting\n");
		err = (*function)(func, gridDim, blockDim, args, sharedMem, stream);
		printf("------------------------ after submitting\n");
	}

	// wait and run
	/* while (true) {
		pthread_mutex_lock(mutexes[0]);
		if (kqueues[0]->front().run) {
			cudaStream_t sched_stream = kqueues[0]->front().sched_stream;
			kqueues[0]->pop();
			printf("-------- run with stream %d!!!\n", sched_stream);
			pthread_mutex_unlock(mutexes[0]);
			err = (*function)(func, gridDim, blockDim, args, sharedMem, sched_stream); 
			return err;
		}
		pthread_mutex_unlock(mutexes[0]);
	} */
}

int main() {

	printf("running......\n");
}
