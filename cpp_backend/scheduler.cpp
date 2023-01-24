#include "scheduler.h"

using namespace std;

void* klib;


void* Scheduler::busy_wait(void** qbuffers, pthread_mutex_t** mutexes, int num_clients) {
	

	printf("entered busy wait!\n");	
			
	queue<struct kernel_record>** buffers = (queue<struct kernel_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		buffers[i] = (queue<struct kernel_record>*)(qbuffers[i]);

	cudaError_t (*function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
	*(void **)(&function) = dlsym(RTLD_DEFAULT, "cudaLaunchKernel");

	cudaStream_t sched_stream;
	cudaStreamCreate(&sched_stream);

	pthread_t mytid = pthread_self();
	printf("--------------- Scheduler id is %d\n", mytid);


	int seen[num_clients] = {0};
	
	int num_kernels = 289;
	int num_iters = 1;
	int it = 0;

	printf("for ID 0: mutex address is %p, buffer address is %p, buffers is %p\n", mutexes[0], buffers[0], buffers);

	while (it < num_iters) {
		for (int i=0; i<num_clients; i++) {
			while (seen[i] < num_kernels) {
				pthread_mutex_lock(mutexes[i]);
				volatile int sz = buffers[i]->size();
				if (sz > 0) {
					//printf("i: %d , sz is: %d\n", i, sz);
					struct kernel_record record = buffers[i]->front();
					
					// case 1
					(*function)(record.func, record.gridDim, record.blockDim, record.args, record.sharedMem, sched_stream);
					cudaDeviceSynchronize();
					buffers[i]->pop();

					// run
					// case 2
					/*if (!record.run) {
					/	buffers[i]->front().sched_stream = sched_stream;
						buffers[i]->front().run = true;   
						seen[i] += 1;*/
						//printf("%d, kernel record func ptr is %p, args is %p, run is %d, stream is %d\n", seen[i], record.func, record.args, record.run, sched_stream);

					//}
				}
				pthread_mutex_unlock(mutexes[i]);
			}

		}
		it += 1;
		for (int i=0; i<num_clients; i++)
			seen[i] = 0;
		printf("restart! %d\n", it);
	}

	return NULL;
	
}

extern "C" {

	Scheduler* sched_init() {
		
		Scheduler* sched = new Scheduler();
		return sched;
	}

	void setup(Scheduler* scheduler, int tid0, int tid1) {

		klib = dlopen("/home/fot/gpu_share/cpp_backend/cuda_capture/libinttemp.so", RTLD_NOW | RTLD_GLOBAL);
		if (!klib) {
			fprintf(stderr, "Error: %s\n", dlerror());
			return;
		}
		
		pthread_t* thread_ids_all = (pthread_t*)dlsym(klib, "thread_ids");
		thread_ids_all[0] = tid0;
		thread_ids_all[1] = tid1;
		
		printf("here!\n");

	}

	void* sched_func(Scheduler* scheduler) { //void* buffer, pthread_mutex_t* mutex) {

		
		//Scheduler* scheduler = (Scheduler*)(arg);
		void** buffers = (void**)dlsym(klib, "kqueues"); 
	
		printf("buffers is %p, %p, %p\n", buffers, buffers[0], buffers[1]);
		pthread_mutex_t** mutexes = (pthread_mutex_t**)dlsym(klib, "mutexes"); 
		int num_clients = 1;

		printf("entered sched func!\n");
		scheduler->busy_wait(buffers, mutexes, num_clients);
		printf("exited sched func!\n");  
		return NULL;
	}
}


