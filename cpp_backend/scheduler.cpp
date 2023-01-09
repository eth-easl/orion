#include "scheduler.h"

using namespace std;

__global__ void my_kernel(int num) {
	printf("------- Num is %d\n", num);
}



void* Scheduler::busy_wait(void* qbuffer, pthread_mutex_t* mutex) {
	

	printf("entered busy wait!\n");	
	queue<struct kernel_record>* buffer = (queue<struct kernel_record>*)qbuffer;
	printf("queue size is %d\n", buffer->size());

	printf("from scheduler: %p\n", buffer);
			
	cudaError_t (*function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
	*(void **)(&function) = dlsym(RTLD_DEFAULT, "cudaLaunchKernel");

	cudaStream_t sched_stream;
	cudaStreamCreate(&sched_stream);

	pthread_t mytid = pthread_self();
	printf("--------------- Scheduler id is %d\n", mytid);

	printf("from scheduler, queue0: %p\n", buffer);

	int seen[1] = {0};
	
	int num_kernels = 289;
	int num_iters = 1;
	int it = 0;

	while(it < num_iters) {
		for (int i=0; i<1; i++) {

			pthread_mutex_lock(mutex);
			volatile int sz = buffer->size();
			if (sz > 0) {
				//printf("i: %d , sz is: %d\n", i, sz);
				struct kernel_record record = buffer->front();
				//buffer->pop();

				// run
				if (!record.run) {
					buffer->front().stream = 0;
					buffer->front().run = true;   
					seen[i] += 1;
					printf("%d, kernel record func ptr is %p, args is %p, run is %d, stream is %d\n", seen[i], record.func, record.args, record.run, sched_stream);

				}
			}
			pthread_mutex_unlock(mutex);

		}
		if (seen[0]==num_kernels) {
			it += 1;
			seen[0] = 0;
			printf("restart! %d\n", it);
		}
	}

	printf("exit!!\n");
	
}

extern "C" {

	Scheduler* sched_init() {
		
		Scheduler* sched = new Scheduler();
		return sched;
	}

	void* sched_func(Scheduler* scheduler) { //void* buffer, pthread_mutex_t* mutex) {

		//Scheduler* scheduler = (Scheduler*)sched;

		void* klib = dlopen("/home/fot/gpu_share/cpp_backend/cuda_capture/libinttemp.so", RTLD_NOW | RTLD_GLOBAL);
		if (!klib) {
			fprintf(stderr, "Error: %s\n", dlerror());				    
	    		return NULL;
		}
		
		void*  buffer = dlsym(klib, "kqueue0"); 
		pthread_mutex_t* mutex = (pthread_mutex_t*)dlsym(klib, "mutex0"); 

		printf("entered sched func!\n");
		scheduler->busy_wait(buffer, mutex);
		return NULL;
	}
}


