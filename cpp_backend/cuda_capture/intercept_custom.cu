#include "intercept_custom.h"

using namespace std;

/*cudaError_t cudaMalloc(void** devPtr, size_t size) {

	printf("Caught cudaMalloc!\n");

	cudaError_t (*function)(void** devPtr, size_t size);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMalloc");
	
	cudaError_t err = (*function)(devPtr, size);
	return err;

}*/

queue<kernel_record> kqueue0;
queue<kernel_record> kqueue1;
pthread_t thread_ids[2];
pthread_mutex_t mutex0;
pthread_mutex_t mutex1;
int i=0;

cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {


	printf("Captured a cudaLaunchKernel! function ptr is %p, stream is %d, gridDim is %d, blockDim is %d, sharedMem is %ld, args is %p\n", func, stream, gridDim, blockDim, sharedMem, args);

	pthread_t mytid = pthread_self();


	cudaError_t (*function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaLaunchKernel");


	cudaError_t err = cudaSuccess;
	if (1) { //stream != 0) {
		printf("------------- RUN!!!!!!!!!!!!!, args is %p\n", args);
		err = (*function)(func, gridDim, blockDim, args, sharedMem, stream);
		printf("------------- %d, %d, %d, after func call!\n", mytid, stream, i);
		//i+=1;
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
