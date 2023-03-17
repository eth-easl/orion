#include "scheduler.h"

using namespace std;

// globals
void* klib;
vector<vector<op_info>> op_info_vector;
int max_sms = 1000; // TODO: set real value

void* Scheduler::busy_wait_fifo(void** qbuffers, pthread_mutex_t** mutexes, int num_clients) {

	DEBUG_PRINT("entered busy wait FIFO!\n");

	register_functions();

	queue<struct func_record>** buffers = (queue<struct func_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		buffers[i] = (queue<struct func_record>*)(qbuffers[i]);


	cudaStream_t sched_stream;
	cudaStreamCreate(&sched_stream);

	cudaEvent_t sched_event;
	cudaEventCreateWithFlags(&sched_event, cudaEventDisableTiming);

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
					schedule_kernel(frecord, sched_stream, i, sched_event);
					buffers[i]->pop();
					seen[i] += 1;
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


void* Scheduler::busy_wait_profile(void** qbuffers, pthread_mutex_t** mutexes, int num_clients) {

	DEBUG_PRINT("entered busy wait PROFILE!\n");

	register_functions();
	queue<struct func_record>** buffers = (queue<struct func_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		buffers[i] = (queue<struct func_record>*)(qbuffers[i]);

	cudaStream_t* lp_stream0 = (cudaStream_t*)malloc(sizeof(cudaStream_t));
	cudaStream_t* lp_stream1 = (cudaStream_t*)malloc(sizeof(cudaStream_t));
	cudaStream_t* hp_stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));

	cudaEvent_t* lp_event0 = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
	cudaEvent_t* lp_event1 = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
	cudaEvent_t* hp_event = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));

	create_streams(lp_stream0, lp_stream1, hp_stream);
	create_events(lp_event0, lp_event1, hp_event);

	int seen[num_clients] = {0};
	int streams[num_clients] = {-1}; // 1: unitialized, 0: low prio, 1: high prio

	int num_kernels = 2000;
	int num_iters = 1;
	int it = 0;

	DEBUG_PRINT("for ID 0: mutex address is %p, buffer address is %p, buffers is %p\n", mutexes[0], buffers[0], buffers);

	DEBUG_PRINT("for ID 1: mutex address is %p, buffer address is %p, buffers is %p\n", mutexes[1], buffers[1], buffers);

	while (it < num_iters) {
		while(1) {
			if (seen[0]==num_kernels and seen[1]==num_kernels)
				break;

			vector<func_record*> frecords = {NULL, NULL};

			for (int i=0; i<num_clients; i++) {
				if (seen[i] == num_kernels)
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
					*lp_stream0,
					*lp_stream1,
					*hp_stream,
					streams,
					*lp_event0,
					*lp_event1,
					*hp_event
				);
			}

			else if (frecords[0] != NULL) {
				wait_for_stream(0, 0, streams[0], *lp_stream0,  *lp_event0,  *lp_event1, *hp_event);
				schedule_kernel(*(frecords[0]), *lp_stream0, 0, *lp_event0);
				streams[0] = 0;
				pop_from_queue(buffers[0], mutexes[0]);
			}

			else if (frecords[1] != NULL) {
				wait_for_stream(1, 0, streams[1], *lp_stream1,  *lp_event0,  *lp_event1, *hp_event);
				schedule_kernel(*(frecords[1]), *lp_stream1, 1, *lp_event1);
				streams[1] = 0;
				pop_from_queue(buffers[1], mutexes[1]);
			}


		}
		it += 1;
		for (int i=0; i<num_clients; i++) {
			seen[i] = 0;
			streams[i] = -1;
		}
		DEBUG_PRINT("restart! %d\n", it);
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

			//std::cout << line << std::endl;
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

		op_info_vector.push_back({});
		op_info_vector.push_back({});

		printf("func_names_all is %p\n", func_names_all);
		printf("fname0 ptr is %p, fname1 ptr is %p\n", func_names_all[0], func_names_all[1]);
		populate_kernel_info(func_names_all[0], file0, op_info_vector[0]);
		printf("----------- SIZE: %d\n", op_info_vector[0].size());
		populate_kernel_info(func_names_all[1], file1, op_info_vector[1]);

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
