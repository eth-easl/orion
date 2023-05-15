#include "scheduler.h"
#define MIN_DURATION 100000 // might need to change this - emperical

using namespace std;

// globals
void* klib;
vector<vector<op_info>> op_info_vector;
int* fidx;
int* num_client_kernels;
int* num_client_max_iters;
int* num_client_cur_iters;
bool* locked;

std::chrono::time_point<std::chrono::high_resolution_clock>* client_starts;
std::chrono::time_point<std::chrono::high_resolution_clock>* total_client_starts;
bool** client_starts_set;
vector<vector<float>> client_durations;

int max_sms = 108; // 80 for V100
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
int status;
vector<int> max_sms_clients;
vector<bool> is_train;

// reef
int lp_idx = 0;
int penalty = 0;
bool** request_status;
bool* stops;
bool* stop_ack;


void Scheduler::profile_reset(int num_clients) {

	for (int i=0; i<num_clients; i++) {
		seen[i] = 0;
		streams[i] = -1;
		fidx[i] = 0;
	}
}

void Scheduler::profile_prep(queue<func_record>** qbuffers, int num_clients, bool reef) {

	printf("Entered profile_prep!\n");

	register_functions();
	client_buffers = (queue<struct func_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		client_buffers[i] = (queue<struct func_record>*)(qbuffers[i]);

	int num = num_clients;

	sched_streams = (cudaStream_t**)malloc((num)*sizeof(cudaStream_t*));
	for (int i=0; i<num; i++)
		sched_streams[i] = NULL;

	events = (cudaEvent_t***)malloc((num)*sizeof(cudaEvent_t**));
	for (int i=0; i<num; i++)
		events[i] = NULL;

	create_streams(sched_streams, num, reef);
	create_events(events, num);

	seen = (int*)calloc(num,sizeof(int));
	event_ids = (int*)calloc(num, sizeof(int));

	streams = (int*)malloc(num_clients*sizeof(int));
	for (int i=0; i<num_clients; i++)
		streams[i] = -1;

	sched_stream = 0;

	status = -1;

	printf("Exited profile_prep!\n");

}


void Scheduler::schedule_reef(vector<func_record*> frecords, int num_clients, int depth) {

	if (num_clients==1) {
		if (frecords[0] != NULL) {
			schedule_kernel(*(frecords[0]), sched_streams[0], 0, events[0][event_ids[0]], seen, event_ids, 0);
			pop_from_queue(client_buffers[0], client_mutexes[0], 0);
		}
		return;
	}


	if (frecords[1] != NULL && frecords[0] == NULL) {
		op_info op_info_1 = op_info_vector[1][seen[1]];
		schedule_kernel(*(frecords[1]), sched_streams[1], 1, events[1][event_ids[1]], seen, event_ids, 1);
		pop_from_queue(client_buffers[1], client_mutexes[1], 1);
		// sync here?
		//CHECK_CUDA_ERROR(cudaStreamSynchronize(*sched_streams[1]));
	}
	else if (frecords[0] != NULL && frecords[1] == NULL) {
		penalty += 1;
		if (penalty == 12) {
			schedule_kernel(*(frecords[0]), sched_streams[0], 0, events[0][event_ids[0]], seen, event_ids, 0);
			pop_from_queue(client_buffers[0], client_mutexes[0], 0);
			if (lp_idx == depth) {
				CHECK_CUDA_ERROR(cudaStreamSynchronize(*sched_streams[0]));
				lp_idx = 0;
			}
			lp_idx += 1;
			penalty = 0;
		}
	}
	if (frecords[0] != NULL && frecords[1] != NULL) {
		op_info op_info_0 = op_info_vector[0][seen[0]];
		op_info op_info_1 = op_info_vector[1][seen[1]];
		schedule_kernel(*(frecords[1]), sched_streams[1], 1, events[1][event_ids[1]], seen, event_ids, 1);
		pop_from_queue(client_buffers[1], client_mutexes[1], 1);
		if (op_info_0.duration < op_info_1.duration && op_info_0.sm_used > op_info_1.sm_used) {
			// colocate
			schedule_kernel(*(frecords[0]), sched_streams[0], 0, events[0][event_ids[0]], seen, event_ids, 0);
			pop_from_queue(client_buffers[0], client_mutexes[0], 0);
		}
		// sync here?
		//CHECK_CUDA_ERROR(cudaStreamSynchronize(*sched_streams[1]));
	}
}

void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, int warmup_iters, bool reef, int depth, int hp_limit, int update_start) {

	printf("here!\n");

	DEBUG_PRINT("Entered busy_wait_profile! Num clients is %d\n", num_clients);
	int start0 = 0;
	int start1 = 0;

	int prev_large = -1;
	int hp_running = -1;

	bool inf_finished = false;
	bool started = false;
 	std::chrono::time_point<std::chrono::system_clock> start_time;
	auto start_total = std::chrono::high_resolution_clock::now();

	vector<bool> total_client_set(num_clients, false);
	vector<int> profiles(num_clients, -1);
	vector<int> cur_sms(num_clients, -1);
	int hp_client = num_clients-1;

	bool large_found = false;
	long sum = 0;

	// BS - works only for 2 clients for now
	int low_sms = 0;
	int high_sms = max_sms_clients[0]; // 0 is the lp client
	int sm_threshold = max_sms_clients[0]/2;
	float hp_iter_duration = 0.0; // 1 is the hp client
	float hp_limit_float = (float)hp_limit;

	// if hp is inference, use max_sms + also there is no update phase
	if (!is_train[1]) {
		sm_threshold = max_sms;
		update_start = INT_MAX;
	}

	while(1) {
		vector<func_record*> frecords(num_clients, NULL);

		for (int i=0; i<num_clients; i++) {
			if (seen[i] == num_client_kernels[i])
				continue;

			pthread_mutex_lock(client_mutexes[i]);
			volatile int sz = client_buffers[i]->size();
			if (sz > 0) {
				frecords[i] = &(client_buffers[i]->front());
				int cur_iter = num_client_cur_iters[i];
				if (seen[i] == 0 && client_starts_set[i][cur_iter] == false) {
					client_starts[i] = std::chrono::high_resolution_clock::now();
					client_starts_set[i][cur_iter] = true;
					if (!total_client_set[i]) {
						total_client_starts[i] = std::chrono::high_resolution_clock::now();
						total_client_set[i] = true;
					}
				}
				//if (seen[i] == num_client_kernels[i]-1)
				//	continue;
			}
			pthread_mutex_unlock(client_mutexes[i]);
		}

		if (reef) {
			schedule_reef(frecords, num_clients, depth);
		}

		else {
			if (frecords[hp_client] != NULL) { // high priority

				op_info op_info_1 = op_info_vector[hp_client][seen[hp_client]];
				schedule_kernel(*(frecords[hp_client]), sched_streams[hp_client], hp_client, events[hp_client][event_ids[hp_client]], seen, event_ids, hp_client);
				streams[hp_client] = 1;
				profiles[hp_client] = op_info_1.profile;
				cur_sms[hp_client] = op_info_1.sm_used;

				status = 1;
				pop_from_queue(client_buffers[hp_client], client_mutexes[hp_client], hp_client);
			}
			for (int j=0; j<num_clients-1; j++) {
				// TODO: update policy here
				if (frecords[j] != NULL) { // low priority
					op_info op_info_0 = op_info_vector[j][seen[j]];
					bool schedule = false;
					bool block = false;

					//printf("%d, %d, %d\n", low_sms, high_sms, sm_threshold);

					if ((num_clients==1) || (seen[hp_client] == 0) || (frecords[j]->type == MALLOC_RECORD) || (frecords[j]->type == MEMCPY_RECORD) || (frecords[j]->type == MEMSET_RECORD) || (frecords[j]->type == FREE_RECORD))
						schedule = true;
					else if (num_client_cur_iters[j] <= 10 || num_client_cur_iters[hp_client] >= num_client_max_iters[hp_client]) {
						// this could be removed
						schedule = true;
					}
					else if (seen[hp_client] >= update_start && (op_info_0.sm_used <= sm_threshold && cudaEventQuery(*(events[hp_client][update_start-1])) == cudaSuccess)) // && (op_info_0.sm_used <= 10*sm_threshold))
						schedule = true;
					else if (seen[hp_client]>0 && (op_info_0.sm_used <= sm_threshold) &&  ((op_info_0.profile == -1 || profiles[hp_client]==-1 || (profiles[hp_client] != op_info_0.profile))))
						schedule = true;
					if (schedule && large_found && event_ids[j]>=1) {
						cudaError_t status = cudaEventQuery(*(events[j][event_ids[j]-1]));
						if (status == cudaSuccess) {
							large_found = false;
							sum = 0;
						}
						else {
							schedule = false;
						}
					}
					if (schedule) {
						//if (op_info_0.duration > depth && num_client_cur_iters[1] < num_client_max_iters[1] && seen[1]==0) {
							//block = true;
						if ((frecords[j]->type != MALLOC_RECORD) && (frecords[j]->type != MEMCPY_RECORD) && (frecords[j]->type != MEMSET_RECORD) && (frecords[j]->type != FREE_RECORD))
							sum += op_info_0.duration;
						if (sum > depth && num_client_cur_iters[hp_client] < num_client_max_iters[hp_client] && seen[hp_client]==0) {
							large_found = true;
						}
						schedule_kernel(*(frecords[j]), sched_streams[j], 0, events[j][event_ids[j]], seen, event_ids, 0);
						status = 0;
						pop_from_queue(client_buffers[j], client_mutexes[j], 0);

						streams[j] = 0;
					}
				}
			}
		}

		int finished = 0;
		for (int i=0; i<num_clients; i++) {
			//printf("%d, %d, %d, %d, %d\n", i, seen[i], num_client_kernels[i], num_client_cur_iters[i], num_client_max_iters[i]);
			//printf("%d, %d\n", is_train[0], is_train[1]);

			if (
				(num_client_cur_iters[i] == num_client_max_iters[i])
				|| (warmup && (num_client_cur_iters[i]==warmup_iters))
				|| (stop_ack[i] == true)
			)
				finished += 1;
			else if (seen[i] == num_client_kernels[i]) {
				// check if GPU work for this client has finished
				if (!locked[i]) {
					pthread_mutex_lock(client_mutexes[i]);
					locked[i] = true;
					DEBUG_PRINT("LOCK CLIENT %d\n", i);
				}
				bool ready = true;
				if (event_ids[i] >= 1) {
					if (cudaEventQuery(*(events[i][event_ids[i]-1])) != cudaSuccess)
						ready &= false;
				}
				if (ready) {
					// if yes, reset meta-structures for this client, and let it continue
					seen[i] = 0;
					event_ids[i] = 0;
					streams[i] = -1;
					fidx[i] = 0;
					request_status[i][num_client_cur_iters[i]] = true;
					pthread_mutex_unlock(client_mutexes[i]);
					num_client_cur_iters[i] += 1;
					locked[i] = false;

					if (i==0) {
						lp_idx = 0;
						penalty = 0;
						sum = 0;
					}

					auto end = std::chrono::high_resolution_clock::now();
					float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - client_starts[i]).count();
					duration /= 1000.0;
					client_durations[i].push_back(duration);
					if (!reef && i==hp_client && is_train[hp_client]) {
						printf("Client %d finished iteration %d, it took %f ms\n", i, num_client_cur_iters[i], duration);
						hp_iter_duration += duration;
						if ((num_client_cur_iters[i] % 10) == 0 && low_sms != sm_threshold) {
							float hp_avg_duration = hp_iter_duration/10.0;
							printf("--------------------- Average iter duration for client 1 is %f ms, limit is %f ms, sm_threshold is %d\n", hp_avg_duration, hp_limit_float, sm_threshold);
							hp_iter_duration = 0;

							// TODO: add better stopping conditions
							if (hp_avg_duration > hp_limit_float) {
								high_sms = sm_threshold;
								sm_threshold = (low_sms+high_sms)/2;
							}
							else {
								low_sms = sm_threshold;
								sm_threshold = (low_sms+high_sms)/2;
							}
						}
					}
					//printf("Client %d finished iteration %d, it took %f ms, seen is %d\n", i, num_client_cur_iters[i], duration, seen[i]);
				}
				if (
					(num_client_cur_iters[i] == num_client_max_iters[i])
					|| (warmup && (num_client_cur_iters[i]==warmup_iters))
					|| (stop_ack[i] == true)
				) {
					finished += 1;
					if (!warmup) {
						auto end_total = std::chrono::high_resolution_clock::now();
						float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - total_client_starts[i]).count();
						duration /= 1000.0;
						printf("Client %d, Total loop took %f sec\n", i, duration);
						if (i==1) {
							printf("======= Client 0 has done %d iterations\n", num_client_cur_iters[0]);
							if (!locked[0])
								pthread_mutex_lock(client_mutexes[0]);
							stops[0] = true;
							if (!locked[0])
								pthread_mutex_unlock(client_mutexes[0]);
						}
					}
				}
			}
		}

		if (finished==num_clients)
			break;

	}
	if (!warmup) {
		auto end_total = std::chrono::high_resolution_clock::now();
		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
		duration /= 1000.0;
		printf("Total loop took %f sec\n", duration);
		//process_eval(client_durations);
	}

	return NULL;
}

extern "C" {

	Scheduler* sched_init() {

		Scheduler* sched = new Scheduler();
		return sched;
	}


	void populate_kernel_info(char* kernel_info_file, vector<op_info> &ops) {

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

			op_info info = {v[0], stoi(v[1]), stoi(v[2]), stoi(v[3]), stof(v[4])};
			ops.push_back(info);
		}

		infile.close();

	}

	void setup_change(Scheduler* scheduler, int client_id, char* file, int num_kernels) {

		// needed for backward

		op_info_vector[client_id].clear();
		populate_kernel_info(file, op_info_vector[client_id]);
		int max_sm_used = 0;
		for (auto info: op_info_vector[client_id])
			max_sm_used = max(max_sm_used, info.sm_used);
		max_sms_clients[client_id] = max_sm_used;
		num_client_kernels[client_id] = num_kernels;

	}

	void setup(
		Scheduler* scheduler,
		int num_clients,
		int* tids,
		char** models,
		char** files,
		int* num_kernels,
		int* num_iters,
		bool* train,
		bool reef
	) {

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

		// 1. thread structures
		pid_t* thread_ids_all = (pid_t*)dlsym(klib, "thread_ids");
		thread_ids_all = (pid_t*)malloc((2*num_clients+1)*sizeof(pid_t)); // 2*N threads + scheduler
		for (int i=0; i<num_clients; i++)
			thread_ids_all[i] = tids[i];
		thread_ids_all[num_clients] = mytid;
		for (int i=num_clients+1; i<2*num_clients+1; i++)
			thread_ids_all[i] = 0;

		int* num_total_clients = (int*)dlsym(klib, "num_total_clients");
		*num_total_clients = num_clients;

		for (int i=0; i<=num_clients; i++) {
			DEBUG_PRINT("Scheduler setup the thread id at %d to be %d\n", i, thread_ids_all[i]);
		}

		// 2. metadata structures
		for (int i=0; i<num_clients; i++) {
			op_info_vector.push_back({});
			client_durations.push_back({});
			populate_kernel_info(files[i], op_info_vector[i]);
			int max_sm_used = 0;
			for (auto info: op_info_vector[i])
				max_sm_used = max(max_sm_used, info.sm_used);
			max_sms_clients.push_back(max_sm_used);
			printf("----------- SIZE: %d\n", op_info_vector[i].size());
			is_train.push_back(train[i]);
		}

		// 3. indexes
		fidx = (int*)dlsym(klib, "func_indexes");
		fidx = (int*)calloc(num_clients, sizeof(int));
		num_client_kernels = num_kernels;
		num_client_max_iters = num_iters;

		num_client_cur_iters = (int*)calloc(num_clients, sizeof(int));
		locked = (bool*)calloc(num_clients, sizeof(bool));

		// to get measurements
		client_starts = (std::chrono::time_point<std::chrono::high_resolution_clock>*)calloc(num_clients, sizeof(std::chrono::time_point<std::chrono::high_resolution_clock>));
		total_client_starts = (std::chrono::time_point<std::chrono::high_resolution_clock>*)calloc(num_clients, sizeof(std::chrono::time_point<std::chrono::high_resolution_clock>));
		client_starts_set = (bool**)malloc(num_clients*sizeof(bool*));
		for (int i=0; i<num_clients; i++) {
			client_starts_set[i] = (bool*)calloc(num_client_max_iters[i], sizeof(bool));
		}

		// 4. communication queues + locks
		queue<func_record>** buffers = (queue<func_record>**)dlsym(klib, "kqueues");
		buffers = (queue<func_record>**)malloc(num_clients*sizeof(queue<func_record>*));
		for (int i=0; i<num_clients; i++) {
			buffers[i] = (queue<func_record>*)malloc(sizeof(queue<func_record>));
		}

		client_mutexes = (pthread_mutex_t**)dlsym(klib, "mutexes");
		client_mutexes = (pthread_mutex_t**)malloc(num_clients*sizeof(pthread_mutex_t*));
		for (int i=0; i<num_clients; i++) {
			client_mutexes[i] = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		}
		scheduler->profile_prep(buffers, num_clients, reef);

		// 5. runtime control
		request_status = (bool**)dlsym(klib, "client_request_status");
		request_status = (bool**)malloc(num_clients*sizeof(bool*));

		// check!
		stops = (bool*)dlsym(klib, "client_stop");
		stops = (bool*)calloc(num_clients, sizeof(bool));

		stop_ack = (bool*)dlsym(klib, "client_stop_ack");
		stop_ack = (bool*)calloc(num_clients, sizeof(bool));

		for (int i=0; i<num_clients; i++) {
			request_status[i] = (bool*)calloc(num_client_max_iters[i], sizeof(bool));
		}

	}


	void* schedule(Scheduler* scheduler, int num_clients, bool profile_mode, int iter, bool warmup, int warmup_iters, bool reef, int reef_depth, int hp_limit, int update_start) {

		DEBUG_PRINT("entered sched func!\n");
		if (profile_mode)
			scheduler->busy_wait_profile(num_clients, iter, warmup, warmup_iters, reef, reef_depth, hp_limit, update_start);
		DEBUG_PRINT("exited sched func!\n");
		return NULL;
	}

	void* reset(Scheduler* scheduler, int num_clients) {
		scheduler->profile_reset(num_clients);
		return NULL;
	}
}