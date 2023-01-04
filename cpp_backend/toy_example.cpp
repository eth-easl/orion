#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <stdarg.h>
#include <cuda_runtime.h>
#include "scheduler.h"
#include <dlfcn.h>
#include <queue>
#include <pthread.h>

using namespace std;

int i = 100;
float j = 17.0;
void* cargs[] = {&i}; 
void* cargs2[] = {&i, &j};  

volatile void* kqueue_struct0;
volatile void* kqueue_struct1;
pthread_barrier_t barrier;

void* app(void* args) {

	// wait
	pthread_t mytid = pthread_self();
	printf("My thread id is %d\n", mytid);

	//pthread_barrier_wait(&barrier);

	if (1) {
		int* arr ;
		cudaError_t cudaStatus = cudaMalloc((void**)(&arr), 10*sizeof(int));
		if (cudaStatus != cudaSuccess) {
			printf("CudaMalloc failed!");
			return NULL;
		}

		printf("arr: %p\n", arr);
		int val = -5;
		void *cuda_args[] = {&arr, &val};
		for (int i=0; i<1; i++) {
			cudaLaunchKernel((void*)toy_kernel_ar, dim3(1), dim3(1), cuda_args, 0, NULL);    
       			//toy_kernel_ar<<<1,1>>>(arr);
		}
		//for (int i=0; i<10; i++) {
		//	cudaLaunchKernel((void*)toy_kernel_two, dim3(1), dim3(1), cargs2, 0, NULL);    
		//}
	}
	while(1) ;

}



int main() {

	cudaSetDevice(0);
	void* klib = dlopen("/home/fot/elastic-spot-ml/scheduling/cpp_backend/cuda_capture/libint.so", RTLD_NOW | RTLD_GLOBAL);
	if (!klib) {
		fprintf(stderr, "Error: %s\n", dlerror());
		return 1;
	}
	
	kqueue_struct0 = dlsym(klib, "kqueue0");
	kqueue_struct1 = dlsym(klib, "kqueue1");

	printf("struct0: %p, struct1: %p\n", kqueue_struct0, kqueue_struct1);

	pthread_mutex_t* mutex0 = (pthread_mutex_t*)dlsym(klib, "mutex0");
	pthread_mutex_t* mutex1 = (pthread_mutex_t*)dlsym(klib, "mutex1");

	pthread_t* tid_ptr = (pthread_t*)dlsym(klib, "thread_ids");

	printf("tid_ptr: %d\n", tid_ptr[0]);

     	pthread_t thread1, thread2, thread_sched;
	Scheduler* sched = sched_init();

	int ret = pthread_barrier_init(&barrier, NULL, 2);
		
	/*struct sched_args sargs;
	sargs.mutexes = (pthread_mutex_t**)malloc(2*sizeof(pthread_mutex_t*));
	sargs.mutexes[0] = mutex0;
	sargs.mutexes[1] = mutex1;

	sargs.barrier = &barrier;
	sargs.buffer = (volatile void**)malloc(2*sizeof(void*));
	sargs.buffer[0] = kqueue_struct0;
	sargs.buffer[1] = kqueue_struct1;*/

	// T1 here - app
	int t1 = pthread_create(&thread1, NULL, app, NULL);
	//int t2 = pthread_create(&thread2, NULL, app, NULL);
	int t3 = pthread_create(&thread_sched, NULL, sched_func, (void*)sched);

	tid_ptr[0] = thread1;
	//tid_ptr[1] = thread2; 

	//printf("threads: %d, %d, %d\n", thread1, thread2, thread_sched);

	pthread_join(thread1, NULL);
	//pthread_join(thread2, NULL); 
	pthread_join(thread_sched, NULL);

	cudaDeviceSynchronize();
	dlclose(klib);
}
