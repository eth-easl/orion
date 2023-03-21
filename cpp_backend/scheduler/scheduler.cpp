#include "scheduler.h"

using namespace std;

// globals
void* klib;
vector<vector<op_info>> op_info_vector;
int* fidx;
int* num_client_kernels;
int max_sms = 1000; // TODO: set real value

void* Scheduler::busy_wait_fifo(void** qbuffers, pthread_mutex_t** mutexes, int num_clients) {

	DEBUG_PRINT("entered busy wait FIFO!\n");

	register_functions();

	queue<struct func_record>** buffers = (queue<struct func_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		buffers[i] = (queue<struct func_record>*)(qbuffers[i]);


	cudaStream_t sched_stream;
	CHECK_CUDA_ERROR(cudaStreamCreate(&sched_stream));

	cudaEvent_t sched_event;
	CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&sched_event, cudaEventDisableTiming));

	int seen[num_clients] = {0};

	int num_iters = 10;
	int it = 0;

	DEBUG_PRINT("for ID 0: mutex address is %p, buffer address is %p, buffers is %p\n", mutexes[0], buffers[0], buffers);

	DEBUG_PRINT("for ID 1: mutex address is %p, buffer address is %p, buffers is %p\n", mutexes[1], buffers[1], buffers);


	while (it < num_iters) {
		while(1) {
			for (int i=0; i<num_clients; i++) {
				if (seen[i] == num_client_kernels[i])
					continue;
				pthread_mutex_lock(mutexes[i]);
				volatile int sz = buffers[i]->size();
				if (sz > 0) {
					struct func_record frecord = buffers[i]->front();
					schedule_kernel(frecord, &sched_stream, i, &sched_event, seen);
					buffers[i]->pop();
				}
				pthread_mutex_unlock(mutexes[i]);
			}

			bool finished = true;
			for (int i=0; i<num_clients; i++) {
				if (seen[i] < num_client_kernels[i]) {
					finished = false;
					break;
				}
			}

			if (finished) {
				break;
			}

		}
		it += 1;
		for (int i=0; i<num_clients; i++) {
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
			seen[i] = 0;
			fidx[i] = 0;
		}
		printf("RESTART! %d\n", it);
	}

	return NULL;
}


void* Scheduler::busy_wait_profile(void** qbuffers, pthread_mutex_t** mutexes, int num_clients) {

	DEBUG_PRINT("entered busy wait PROFILE!\n");

	register_functions();
	queue<struct func_record>** buffers = (queue<struct func_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		buffers[i] = (queue<struct func_record>*)(qbuffers[i]);

	cudaStream_t* sched_streams[num_clients+1] = {NULL};
	cudaEvent_t* events[num_clients+1] = {NULL};

	create_streams(sched_streams, num_clients+1);
	create_events(events, num_clients+1);

	int seen[num_clients] = {0};
	int streams[num_clients] = {-1}; // 1: unitialized, 0: low prio, 1: high prio
	int num_iters = 10;
	int it = 0;

	DEBUG_PRINT("for ID 0: mutex address is %p, buffer address is %p, buffers is %p\n", mutexes[0], buffers[0], buffers);

	DEBUG_PRINT("for ID 1: mutex address is %p, buffer address is %p, buffers is %p\n", mutexes[1], buffers[1], buffers);

	while (it < num_iters) {
		while(1) {
			vector<func_record*> frecords = {NULL, NULL};

			for (int i=0; i<num_clients; i++) {
				if (seen[i] == num_client_kernels[i])
					continue;
				pthread_mutex_lock(mutexes[i]);
				volatile int sz = buffers[i]->size();
				if (sz > 0) {
					frecords[i] = &(buffers[i]->front());
				}
				pthread_mutex_unlock(mutexes[i]);
			}

			if ((frecords[0] != NULL) and (frecords[1] != NULL)) {
				schedule_pair(
					frecords,
					buffers,
					mutexes,
					op_info_vector,
					seen,
					max_sms,
					sched_streams,
					streams,
					events,
					num_clients+1
				);
			}

			else if (frecords[0] != NULL) {
				wait_for_stream(0, 0, streams[0], sched_streams[0],  events, num_clients+1);
				schedule_kernel(*(frecords[0]), sched_streams[0], 0, events[0], seen);
				streams[0] = 0;
				pop_from_queue(buffers[0], mutexes[0]);
			}

			else if (frecords[1] != NULL) {
				wait_for_stream(1, 0, streams[1], sched_streams[1],  events, num_clients+1);
				schedule_kernel(*(frecords[1]), sched_streams[1], 1, events[1], seen);
				streams[1] = 0;
				pop_from_queue(buffers[1], mutexes[1]);
			}

			bool finished = true;
			for (int i=0; i<num_clients; i++) {
				if (seen[i] < num_client_kernels[i]) {
					finished = false;
					break;
				}
			}

			if (finished) {
				break;
			}

		}
		it += 1;
		for (int i=0; i<num_clients; i++) {
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
			seen[i] = 0;
			streams[i] = -1;
			fidx[i] = 0;
		}
		printf("RESTART! %d\n", it);
	}

	return NULL;
}

extern "C" {

	Scheduler* sched_init() {

		Scheduler* sched = new Scheduler();
		return sched;
	}


	void populate_kernel_info(vector<char*>* kernel_vector, char* kernel_info_file, vector<op_info> &ops) {

		// TODO: make this more generic, e.g. pass files/models w.r.t input
		printf("KERNEL_INFO_FILE IS %s\n", kernel_info_file);
		string line;
		std::ifstream infile(kernel_info_file);
		assert (infile.is_open());

		// ignore header
		std::getline(infile, line);

		while (std::getline(infile, line))
		{

			std::cout << line << std::endl;
			vector<string> v;
			stringstream sline = stringstream(line);
			while (sline.good()) {
        		string substr;
        		getline(sline, substr, ',');
        		v.push_back(substr);
    		}

			char* kernel_name = new char[v[0].length()+1];
			strcpy(kernel_name, v[0].c_str());
			kernel_vector->push_back(kernel_name);

			op_info info = {v[0], stoi(v[1]), stoi(v[2]), stoi(v[3]), stof(v[4])};
			ops.push_back(info);
		}

		infile.close();

	}


	void setup(Scheduler* scheduler, int num_clients, int* tids, char** models, char** files, int* num_kernels) {

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
		for (int i=0; i<num_clients; i++)
			thread_ids_all[i] = tids[i];
		thread_ids_all[2] = mytid; // TODO: make this configurable

		DEBUG_PRINT("Scheduler setup the thread ids to be %d, %d, %d\n", thread_ids_all[0], thread_ids_all[1], thread_ids_all[2]);

		vector<char*>** func_names_all = (vector<char*>**)dlsym(klib, "func_names");

		char** model_names_all = (char**)dlsym(klib, "model_names");
		for (int i=0; i<num_clients; i++) {
			model_names_all[i] = models[i];
		}

		for (int i=0; i<num_clients; i++) {
			op_info_vector.push_back({});
			printf("fname0 ptr is %p, fname1 ptr is %p\n", func_names_all[i], func_names_all[i]);
			populate_kernel_info(func_names_all[i], files[i], op_info_vector[i]);
			printf("----------- SIZE: %d\n", op_info_vector[i].size());
		}

		fidx = (int*)dlsym(klib, "func_indexes");
		num_client_kernels = num_kernels;

	}

	void* sched_func(Scheduler* scheduler, int num_clients, bool profile_mode) {

		//Scheduler* scheduler = (Scheduler*)(arg);
		void** buffers = (void**)dlsym(klib, "kqueues");

		DEBUG_PRINT("buffers is %p, %p, %p\n", buffers, buffers[0], buffers[1]);
		pthread_mutex_t** mutexes = (pthread_mutex_t**)dlsym(klib, "mutexes");

		DEBUG_PRINT("entered sched func!\n");

		if (profile_mode)
			scheduler->busy_wait_profile(buffers, mutexes, num_clients);
		else
			scheduler->busy_wait_fifo(buffers, mutexes, num_clients);
		DEBUG_PRINT("exited sched func!\n");
		return NULL;
	}
}
