#include "scheduler.h"

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
vector<vector<float>> client_durations;

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
int status;

// new
int* client_ids;
int** shmem_addr;
int** shmem_streams_addr;

using namespace boost::interprocess;

mapped_region* region0;
mapped_region* region1;

mapped_region* streams_region0;
mapped_region* streams_region1;

void* Scheduler::busy_wait_fifo(int num_clients) {
    return NULL;
}

void* Scheduler::busy_wait_single_client(int client_id) {
    return NULL;
}


void Scheduler::schedule_pair(vector<int> frecords) {

	op_info op_info_0 = op_info_vector[0][seen[0]];
	op_info op_info_1 = op_info_vector[1][seen[1]];

	if ((frecords[0] == MALLOC_RECORD) || (frecords[0] == FREE_RECORD) || (frecords[0] == MEMCPY_RECORD) || (frecords[0] == MEMSET_RECORD)) {
		schedule_op(frecords[0], 0, 0);
	}
	else if ((frecords[1] == MALLOC_RECORD) || (frecords[1] == FREE_RECORD) || (frecords[1] == MEMCPY_RECORD) || (frecords[1] == MEMSET_RECORD)) {
		schedule_op(frecords[1], 1, 0);
	}
	else if (op_info_0.duration < 10000.0) {
		schedule_op(frecords[0], 0, 1);
	}
	else if (op_info_1.duration < 10000.0) {
		schedule_op(frecords[1], 1, 1);
	}
	else if (op_info_0.sm_used < max_sms && op_info_1.sm_used < max_sms) {
		//printf("found pair!\n");
		schedule_op(frecords[0], 0, 0);
		schedule_op(frecords[1], 1, 0);
	}
	else if (op_info_0.profile > -1 && (op_info_0.profile == op_info_1.profile)) {
		schedule_op(frecords[0], 0, 0);
	}
	else if (op_info_0.sm_used >= max_sms && op_info_1.sm_used < max_sms) {
		//printf("found pair!\n");
		schedule_op(frecords[0], 0, 0);
		schedule_op(frecords[1], 1, 1);
	}
	else if (op_info_0.sm_used < max_sms && op_info_1.sm_used >= max_sms) {
		//printf("found pair!\n");
		schedule_op(frecords[0], 0, 1);
		schedule_op(frecords[1], 1, 0);
	}
	else {
		schedule_op(frecords[0], 0, 0);
		schedule_op(frecords[1], 1, 0);
	}
}


void Scheduler::schedule_op(int op_type, int idx, int priority) {
	if ((op_type != MALLOC_RECORD) && (op_type != FREE_RECORD) && (op_type != MEMCPY_RECORD) && (op_type != MEMSET_RECORD)) {
		seen[idx]++;
	}
	if ((op_type != MALLOC_RECORD) && (op_type != FREE_RECORD)) {
		*(shmem_streams_addr[idx]) = priority;
	}
	*(shmem_addr[idx]) = -1;
}

void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, bool reef) {

	printf("Enter busy_wait_profile!\n");
    while(1) {

		vector<int> frecords = {-1, -1};
		for (int i=0; i<num_clients; i++) {
			if (seen[i] == num_client_kernels[i])
				continue;

			volatile int *status = shmem_addr[i];
			if (*status >= 0) {
				frecords[i] = *status;
			}
		}
		//if (frecords[0] >= 0 && frecords[1] >= 0) {
		//	schedule_pair(frecords);
		//}
		if (frecords[0] >= 0) {
			schedule_op(frecords[0], 0, 0);
		}
		else if (frecords[1] >= 0) {
			schedule_op(frecords[1], 1, 1);
		}

		int finished = 0;
		for (int i=0; i<num_clients; i++) {
			if (num_client_cur_iters[i] == num_client_max_iters[i]) {
				//if (i==0)
					//printf("Client %d has finished!\n", i);
				finished += 1;
			}
			else if (seen[i] == num_client_kernels[i]) {
				seen[i] = 0;
				num_client_cur_iters[i] += 1;
				//printf("Client %d has done %d iters\n", i, num_client_cur_iters[i]);
			}
		}
		if (finished==num_clients)
			break;
    }
    return NULL;
}

void Scheduler::profile_reset(int num_clients) {
    return;
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
		num_client_kernels[client_id] = num_kernels;

	}

	void setup(
		Scheduler* scheduler,
		int num_clients,
		int* tids,
		char** models,
		char** files,
		int* num_kernels,
		int* num_iters
	) {

        client_ids = (int*)malloc(num_clients*sizeof(int));
        shmem_addr = (int**)malloc(num_clients*sizeof(int*));
		shmem_streams_addr = (int**)malloc(num_clients*sizeof(int*));

        for (int i=0; i<num_clients; i++) {
            client_ids[i] = tids[i];
            std::string shmem_string_name = "client" + std::to_string(tids[i]);
            const char* shmem_name = shmem_string_name.c_str();
            shared_memory_object shm (create_only, shmem_name, read_write);
            shm.truncate(4);

			std::string shmem_string_name_streams = "client_streams" + std::to_string(tids[i]);
            const char* shmem_name_streams = shmem_string_name_streams.c_str();
            shared_memory_object shm_streams (create_only, shmem_name_streams, read_write);
            shm_streams.truncate(4);

			if (i==0) {
				region0 = new mapped_region(shm, read_write);
            	std::memset(region0->get_address(), -1, region0->get_size());
            	shmem_addr[i] = (int*)(region0->get_address());

				streams_region0 = new mapped_region(shm_streams, read_write);
            	std::memset(streams_region0->get_address(), -1, streams_region0->get_size());
            	shmem_streams_addr[i] = (int*)(streams_region0->get_address());
			}
            else {
				region1 = new mapped_region(shm, read_write);
            	std::memset(region1->get_address(), -1, region1->get_size());
            	shmem_addr[i] = (int*)(region1->get_address());

				streams_region1 = new mapped_region(shm_streams, read_write);
            	std::memset(streams_region1->get_address(), -1, streams_region1->get_size());
            	shmem_streams_addr[i] = (int*)(streams_region1->get_address());
			}
            // key_t key = ftok(shmem_name,65);

    		// // shmget returns an identifier in shmid
    		// int shmid = shmget(key,1024,0666|IPC_CREAT);
			// // shmat to attach to shared memory
   			// char *str = (char*) shmat(shmid,(void*)0,0);
			// shmem_addr[i] = (int*)str;
			printf("------------- %d, region %s mapped at address %p, set to %d\n", i, shmem_name, shmem_addr[i], *(shmem_addr[i]));
			printf("------------- %d, region %s mapped at address %p, set to %d\n", i, shmem_name_streams, shmem_streams_addr[i], *(shmem_streams_addr[i]));

			op_info_vector.push_back({});
			populate_kernel_info(files[i], op_info_vector[i]);

        }

		num_client_kernels = num_kernels;
		num_client_max_iters = num_iters;
		num_client_cur_iters = (int*)calloc(num_clients, sizeof(int));
		seen = (int*)calloc(num_clients, sizeof(int));

		return;

	}

	void* sched_setup(Scheduler* scheduler, int num_clients, bool profile_mode, bool reef) {

		return NULL;
	}


	void* schedule(Scheduler* scheduler, int num_clients, bool profile_mode, int iter, bool warmup, bool reef) {

		DEBUG_PRINT("entered sched func!\n");
		if (profile_mode)
			scheduler->busy_wait_profile(num_clients, iter, warmup, reef);
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
