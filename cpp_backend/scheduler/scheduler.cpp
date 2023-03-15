#include "scheduler.h"

using namespace std;

void* klib;


void* Scheduler::busy_wait(void** qbuffers, pthread_mutex_t** mutexes, int num_clients) {



	DEBUG_PRINT("entered busy wait!\n");

	register_functions();

	queue<struct func_record>** buffers = (queue<struct func_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		buffers[i] = (queue<struct func_record>*)(qbuffers[i]);



	cudaStream_t sched_stream;
	cudaStreamCreate(&sched_stream);

	int seen[num_clients] = {0};

	int num_kernels = 1000;
	int num_iters = 1;
	int it = 0;

	DEBUG_PRINT("for ID 0: mutex address is %p, buffer address is %p, buffers is %p\n", mutexes[0], buffers[0], buffers);

	DEBUG_PRINT("for ID 1: mutex address is %p, buffer address is %p, buffers is %p\n", mutexes[1], buffers[1], buffers);


	while (it < num_iters) {
		while(1) {
			if (seen[0]==num_kernels and seen[1]==num_kernels)
				break;

			for (int i=0; i<num_clients; i++) {
				if (seen[i] == num_kernels)
					continue;
				pthread_mutex_lock(mutexes[i]);
				volatile int sz = buffers[i]->size();
				if (sz > 0) {
					struct func_record frecord = buffers[i]->front();
					printf("found a record!\n");
					schedule_kernel(frecord, sched_stream);
					buffers[i]->pop();

				}
				pthread_mutex_unlock(mutexes[i]);
			}

		}
		it += 1;
		for (int i=0; i<num_clients; i++)
			seen[i] = 0;
		DEBUG_PRINT("restart! %d\n", it);
	}

	return NULL;

}

extern "C" {

	Scheduler* sched_init() {

		Scheduler* sched = new Scheduler();
		return sched;
	}


	void populate_kernel_names(vector<char*>* kernel_vector, char* kernel_info_file) {

		// TODO: make this more generic, e.g. pass files/models w.r.t input
		printf("KERNEL_INFO_FILE IS %s\n", kernel_info_file);
		string line;
		std::ifstream infile(kernel_info_file);
		assert (infile.is_open());
		while (std::getline(infile, line))
		{
			char* kernel_name = new char[line.length()+1];
			strcpy(kernel_name, line.c_str());
			kernel_vector->push_back(kernel_name);
		}

		infile.close();

	}


	void setup(Scheduler* scheduler, int tid0, int tid1, char* model0, char* file0, char* model1, char* file1) {

		struct passwd *pw = getpwuid(getuid());
		char *homedir = pw->pw_dir;
		char* lib_path = "/gpu_share_repo/cpp_backend/cuda_capture/libinttemp.so";

		klib = dlopen(strcat(homedir, lib_path), RTLD_NOW | RTLD_GLOBAL);

		if (!klib) {
			fprintf(stderr, "Error: %s\n", dlerror());
			return;
		}

#ifdef SYS_gettid
		pid_t mytid = syscall(SYS_gettid);
#else
#error "SYS_gettid unavailable on this system"
#endif

		pid_t* thread_ids_all = (pid_t*)dlsym(klib, "thread_ids");
		thread_ids_all[0] = tid0;
		thread_ids_all[1] = tid1;
		thread_ids_all[2] = mytid;

		DEBUG_PRINT("Scheduler setup the thread ids to be %d, %d, %d\n", thread_ids_all[0], thread_ids_all[1], thread_ids_all[2]);


		int num_kernels = 1;
		vector<char*>** func_names_all = (vector<char*>**)dlsym(klib, "func_names");

		char** model_names_all = (char**)dlsym(klib, "model_names");
		model_names_all[0] = model0;
		model_names_all[1] = model1;

		printf("func_names_all is %p\n", func_names_all);
		printf("fname0 ptr is %p, fname1 ptr is %p\n", func_names_all[0], func_names_all[1]);
		populate_kernel_names(func_names_all[0], file0);
		populate_kernel_names(func_names_all[1], file1);

	}

	void* sched_func(Scheduler* scheduler, int num_clients) {

		//Scheduler* scheduler = (Scheduler*)(arg);
		void** buffers = (void**)dlsym(klib, "kqueues");

		DEBUG_PRINT("buffers is %p, %p, %p\n", buffers, buffers[0], buffers[1]);
		pthread_mutex_t** mutexes = (pthread_mutex_t**)dlsym(klib, "mutexes");

		DEBUG_PRINT("entered sched func!\n");
		scheduler->busy_wait(buffers, mutexes, num_clients);
		DEBUG_PRINT("exited sched func!\n");
		return NULL;
	}
}
