#include "scheduler.h"

using namespace std;

// globals
void* klib;
vector<vector<op_info>> op_info_vector;
int* fidx;
int* num_client_kernels;
int max_sms = 80;
queue<struct func_record>** client_buffers;
pthread_mutex_t** client_mutexes;
queue<struct func_record>** buffers;
int* seen;
cudnnHandle_t* global_handle0;
cudnnHandle_t* global_handle1;

// fifo-globals
cudaStream_t sched_stream;
cudaEvent_t sched_event;

// profile-globals
cudaStream_t** sched_streams;
cudaEvent_t*** events;
int* streams;
int* event_ids;

void  Scheduler::fifo_prep(void** qbuffers, int num_clients) {

	register_functions();
	client_buffers = (queue<struct func_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		client_buffers[i] = (queue<struct func_record>*)(qbuffers[i]);

	//CHECK_CUDA_ERROR(cudaStreamCreate(&sched_stream));
	sched_stream = 0;
	CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&sched_event, cudaEventDisableTiming));
	seen = (int*)calloc(num_clients,sizeof(int));
	event_ids = (int*)calloc(num_clients+1, sizeof(int));


}

void Scheduler::profile_reset(int num_clients) {

	for (int i=0; i<num_clients; i++) {
		seen[i] = 0;
		streams[i] = -1;
		fidx[i] = 0;
	}
}

void Scheduler::profile_prep(void** qbuffers, int num_clients) {

	register_functions();
	client_buffers = (queue<struct func_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		client_buffers[i] = (queue<struct func_record>*)(qbuffers[i]);

	sched_streams = (cudaStream_t**)malloc((num_clients+1)*sizeof(cudaStream_t*));
	for (int i=0; i<=num_clients; i++)
		sched_streams[i] = NULL;

	events = (cudaEvent_t***)malloc((num_clients+1)*sizeof(cudaEvent_t**));
	for (int i=0; i<=num_clients; i++)
		events[i] = NULL;

	create_streams(sched_streams, num_clients+1);
	create_events(events, num_clients+1);

	seen = (int*)calloc(num_clients,sizeof(int));
	event_ids = (int*)calloc(num_clients+1, sizeof(int));

	streams = (int*)malloc(num_clients*sizeof(int));
	for (int i=0; i<num_clients; i++)
		streams[i] = -1;

	sched_stream = 0;

}

void* Scheduler::busy_wait_fifo(int num_clients) {

	while(1) {
		for (int i=0; i<num_clients; i++) {
			if (seen[i] == num_client_kernels[i])
				continue;
			pthread_mutex_lock(client_mutexes[i]);
			volatile int sz = client_buffers[i]->size();
			if (sz > 0) {
				struct func_record frecord = client_buffers[i]->front();
				schedule_kernel(frecord, &sched_stream, i, &sched_event, seen, event_ids, 0);
				client_buffers[i]->pop();
			}
			pthread_mutex_unlock(client_mutexes[i]);
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

	for (int i=0; i<num_clients; i++) {
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		seen[i] = 0;
		fidx[i] = 0;
	}
	printf("RESTART!\n");

	return NULL;
}

void* Scheduler::busy_wait_single_client(int client_id) {

	printf("Inside busy_wait_single_client for client %d\n", client_id);
	while(seen[client_id] < num_client_kernels[client_id]) {
		pthread_mutex_lock(client_mutexes[client_id]);
		volatile int sz = client_buffers[client_id]->size();
		if (sz > 0) {
			struct func_record frecord = client_buffers[client_id]->front();
			schedule_kernel(frecord, &sched_stream, client_id, &sched_event, seen, event_ids, 0);
			client_buffers[client_id]->pop();
		}
		pthread_mutex_unlock(client_mutexes[client_id]);
	}

	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	seen[client_id] = 0;
	fidx[client_id] = 0;

	printf("RESTART!\n");

	return NULL;
}

void* Scheduler::busy_wait_profile(int num_clients, int iter) {

	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
	int start0 = 0;
	int start1 = 0;

	int prev_large = -1;
	int hp_running = -1;

	cudaEvent_t** events0 = (cudaEvent_t**)malloc(num_client_kernels[0]*sizeof(cudaEvent_t));
	cudaEvent_t** events1 = (cudaEvent_t**)malloc(num_client_kernels[1]*sizeof(cudaEvent_t));

	for (int i=0; i<num_client_kernels[0]; i++) {
		events0[i] = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
		CHECK_CUDA_ERROR(cudaEventCreateWithFlags(events0[i], cudaEventDisableTiming));
	}

	for (int i=0; i<num_client_kernels[1]; i++) {
		events1[i] = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
		CHECK_CUDA_ERROR(cudaEventCreateWithFlags(events1[i], cudaEventDisableTiming));
	}
	if (1) {

		while(1) {
			vector<func_record*> frecords = {NULL, NULL};

			for (int i=0; i<num_clients; i++) {
				if (seen[i] == num_client_kernels[i])
					continue;

				if (iter>=0) {
					pthread_mutex_lock(client_mutexes[i]);
					volatile int sz = client_buffers[i]->size();
					if (sz > 0) {
						frecords[i] = &(client_buffers[i]->front());
					}
					pthread_mutex_unlock(client_mutexes[i]);
				}

				else {
					while(1) {
						pthread_mutex_lock(client_mutexes[i]);
						volatile int sz = client_buffers[i]->size();
						if (sz > 0) {
							frecords[i] = &(client_buffers[i]->front());
							pthread_mutex_unlock(client_mutexes[i]);
							break;
						}
						pthread_mutex_unlock(client_mutexes[i]);
					}
				}

			}

			// if (frecords[0] != NULL && ((frecords[0]->type == MALLOC_RECORD) || (frecords[0]->type == MEMCPY_RECORD) || (frecords[0]->type == FREE_RECORD))) {
			// 	printf("0, iter: %d, type: %d\n", iter, frecords[0]->type);
			// }

			// if (frecords[1] != NULL && ((frecords[1]->type == MALLOC_RECORD) || (frecords[1]->type == MEMCPY_RECORD) || (frecords[1]->type == FREE_RECORD))) {
			// 	printf("1, iter: %d, type: %d\n", iter, frecords[1]->type);
			// }

			//printf("frecords[0]=%p, frecords[1]=%p\n", frecords[0], frecords[1]);

			// auto start = std::chrono::high_resolution_clock::now();
			// auto elapsed = std::chrono::high_resolution_clock::now() - start;
			// long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

			// printf("Elapsed time is %ld\n", microseconds);

			// if ((frecords[0] != NULL) and (frecords[1] != NULL)) {
			// 	//printf("type0 is %d, type1 is %d\n", frecords[0]->type, frecords[1]->type);
			// 	schedule_pair(
			// 		frecords,
			// 		client_buffers,
			// 		client_mutexes,
			// 		op_info_vector,
			// 		seen,
			// 		max_sms,
			// 		sched_streams,
			// 		streams,
			// 		events,
			// 		num_clients+1,
			// 		event_ids
			// 	);
			// }

			// if (frecords[0] != NULL)
			// 	printf("0, %d, %d, %d\n", seen[0], frecords[0]->type, prev_large);

			// if (frecords[1] != NULL)
			// 	printf("1, %d, %d, %d\n", seen[1], frecords[1]->type, prev_large);

			if (prev_large == 0) {
				if (frecords[1] != NULL) {
					op_info op_info_1 = op_info_vector[1][seen[1]];
					if (op_info_1.duration < 10000.0 || op_info_1.sm_used < max_sms){
						//printf("1. Run 1, %d, prev_large is %d\n", seen[1], prev_large);
						wait_for_stream(1, 1, streams[1], sched_streams[num_clients], events, num_clients+1, event_ids);
						// if (op_info_1.duration > 10000.0 && op_info_1.sm_used >= 16)
						// 	sleep_kernel(10000, *(sched_streams[num_clients]));
						schedule_kernel(*(frecords[1]), sched_streams[num_clients], 1, events[num_clients][event_ids[num_clients]], seen, event_ids, num_clients);
						streams[1] = 1;
						hp_running = op_info_1.profile;
						pop_from_queue(client_buffers[1], client_mutexes[1]);
						continue;
					}
				}
			}

			if (prev_large == 1) {
				if (frecords[0] != NULL) {
					op_info op_info_0 = op_info_vector[0][seen[0]];
					if (op_info_0.duration < 10000.0 || op_info_0.sm_used < max_sms) {
						//printf("1. Run 0, %d, prev_large is %d\n", seen[0], prev_large);
						wait_for_stream(0, 1, streams[0], sched_streams[num_clients], events, num_clients+1, event_ids);
						// if (op_info_0.duration > 10000.0 && op_info_0.sm_used >= 16)
						// 	sleep_kernel(10000, *(sched_streams[num_clients]));
						schedule_kernel(*(frecords[0]), sched_streams[num_clients], 0, events[num_clients][event_ids[num_clients]], seen, event_ids, num_clients);
						streams[0] = 1;
						hp_running = op_info_0.profile;
						pop_from_queue(client_buffers[0], client_mutexes[0]);
						continue;
					}
				}

			}
			if (frecords[0] != NULL) {
				//wait_all_streams(0, sched_streams[0], events, num_clients+1, event_ids);

				//printf("record 0 is %d, have seen %d kernels\n", frecords[0]->type, seen[0]);
				op_info op_info_0 = op_info_vector[0][seen[0]];
				if (op_info_0.duration < 10000.0 || op_info_0.sm_used < max_sms) {
					//if (!(iter >= 3 && prev_large==0 &&  seen[1] < num_client_kernels[0])) {
						//printf("2. Run 0, %d, prev_large is %d\n", seen[0], prev_large);
						wait_for_stream(0, 1, streams[0], sched_streams[num_clients], events, num_clients+1, event_ids);
						//if (op_info_0.duration > 10000.0)
						//	sleep_kernel(100000, *(sched_streams[num_clients]));
						schedule_kernel(*(frecords[0]), sched_streams[num_clients], 0, events[num_clients][event_ids[num_clients]], seen, event_ids, num_clients);
						streams[0] = 1;
						hp_running = op_info_0.profile;
						pop_from_queue(client_buffers[0], client_mutexes[0]);
					//}
				}
				else {
					//printf("3. Run 0, %d, prev_large is %d\n", seen[0], prev_large);
					wait_for_stream(0, 0, streams[0], sched_streams[0],  events, num_clients+1, event_ids);
					// if (hp_running == op_info_0.profile)
					//  	CHECK_CUDA_ERROR(cudaStreamWaitEvent(*(sched_streams[0]), *(events[num_clients][event_ids[num_clients]-1]), 0));
					schedule_kernel(*(frecords[0]), sched_streams[0], 0, events[0][event_ids[0]], seen, event_ids, 0);
					streams[0] = 0;
					prev_large = 0;
					pop_from_queue(client_buffers[0], client_mutexes[0]);
					continue;
				}
			}
			if (frecords[1] != NULL) {
				//wait_all_streams(1, sched_streams[1], events, num_clients+1, event_ids);
				//printf("record 1 is %d, have seen %d kernels\n", frecords[1]->type, seen[1]);
				op_info op_info_1 = op_info_vector[1][seen[1]];
				if (op_info_1.duration < 10000.0 || op_info_1.sm_used < max_sms){
					//if (!(iter >= 3 && prev_large==1 &&  seen[0] < num_client_kernels[0])) {
						//printf("2. Run 1, %d, prev_large is %d\n", seen[1], prev_large);
						wait_for_stream(1, 1, streams[1], sched_streams[num_clients], events, num_clients+1, event_ids);
						//if (op_info_1.duration > 10000.0)
						//	sleep_kernel(100000, *(sched_streams[num_clients]));
						schedule_kernel(*(frecords[1]), sched_streams[num_clients], 1, events[num_clients][event_ids[num_clients]], seen, event_ids, num_clients);
						streams[1] = 1;
						hp_running = op_info_1.profile;
						pop_from_queue(client_buffers[1], client_mutexes[1]);
					//}
				}
				else {
					//printf("3. Run 1, %d, prev_large is %d\n", seen[1], prev_large);
					wait_for_stream(1, 0, streams[1], sched_streams[1],  events, num_clients+1, event_ids);
					// if (hp_running == op_info_1.profile)
					//  	CHECK_CUDA_ERROR(cudaStreamWaitEvent(*(sched_streams[1]), *(events[num_clients][event_ids[num_clients]-1]), 0));
					schedule_kernel(*(frecords[1]), sched_streams[1], 1, events[1][event_ids[1]], seen, event_ids, 1);
					streams[1] = 0;
					prev_large = 1;
					pop_from_queue(client_buffers[1], client_mutexes[1]);
				}
			}

			bool finished = true;
			for (int i=0; i<num_clients; i++) {
				//DEBUG_PRINT("%d, %d\n", seen[0], seen[1]);
				if (seen[i] < num_client_kernels[i]) {
					finished = false;
					break;
				}
			}

			if (finished) {
				break;
			}

		}
		for (int i=0; i<num_clients; i++) {
			seen[i] = 0;
			streams[i] = -1;
			fidx[i] = 0;
			event_ids[i] = 0;
		}
		event_ids[num_clients] = 0;
		prev_large = -1;
		hp_running = -1;
		//create_events(events, num_clients+1);
		//printf("RESTART!\n");
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

	void* sched_setup(Scheduler* scheduler, int num_clients, bool profile_mode) {

		//Scheduler* scheduler = (Scheduler*)(arg);
		void** buffers = (void**)dlsym(klib, "kqueues");
		global_handle0 = NULL;
		global_handle1 = NULL;

		DEBUG_PRINT("buffers is %p, %p, %p\n", buffers, buffers[0], buffers[1]);
		client_mutexes = (pthread_mutex_t**)dlsym(klib, "mutexes");

		DEBUG_PRINT("entered sched setup func!\n");

		if (profile_mode)
			scheduler->profile_prep(buffers, num_clients);
		else
			scheduler->fifo_prep(buffers, num_clients);
		DEBUG_PRINT("exited sched prep func!\n");

		return NULL;
	}


	void* schedule(Scheduler* scheduler, int num_clients, bool profile_mode, int iter) {

		DEBUG_PRINT("entered sched func!\n");
		if (profile_mode)
			scheduler->busy_wait_profile(num_clients, iter);
		else
			scheduler->busy_wait_fifo(num_clients);
		DEBUG_PRINT("exited sched func!\n");
		return NULL;
	}

	void* schedule_one(Scheduler* scheduler, int client_id) {
		scheduler->busy_wait_single_client(client_id);
		return NULL;
	}

	void* reset(Scheduler* scheduler, int num_clients) {
		scheduler->profile_reset(num_clients);
		return NULL;
	}
}
