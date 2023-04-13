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

using namespace boost::interprocess;

mapped_region* region0;
mapped_region* region1;

void* Scheduler::busy_wait_fifo(int num_clients) {
    return NULL;
}

void* Scheduler::busy_wait_single_client(int client_id) {
    return NULL;
}

void* Scheduler::busy_wait_profile(int num_clients, int iter, bool warmup, bool reef) {

	printf("Enter busy_wait_profile!\n");
    printf("status 1 is %p, %d!\n", shmem_addr[0], *(shmem_addr[0]));
    while(1) {
        volatile int *status = shmem_addr[0];
        if (*status >= 0) {
		    //printf("HELLO! status is %d!\n", *(shmem_addr[0]));
			*status = -1;
		}
		volatile int *status1 = shmem_addr[1];
        if (*status1 >= 0) {
		    //printf("HELLO! status is %d!\n", *(shmem_addr[0]));
			*status1 = -1;
		}
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

	void setup_change(Scheduler* scheduler, int client_id, char* file, int num_kernels) {

		// needed for backward
		vector<char*>** func_names_all = (vector<char*>**)dlsym(klib, "func_names");
		(*func_names_all[client_id]).clear();
		op_info_vector[client_id].clear();
		populate_kernel_info(func_names_all[client_id], file, op_info_vector[client_id]);
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
        for (int i=0; i<num_clients; i++) {
            client_ids[i] = tids[i];
            std::string shmem_string_name = "client" + std::to_string(tids[i]);
            const char* shmem_name = shmem_string_name.c_str();
            shared_memory_object shm (create_only, shmem_name, read_write);
            shm.truncate(4);
			if (i==0) {
				region0 = new mapped_region(shm, read_write);
            	std::memset(region0->get_address(), -1, region0->get_size());
            	shmem_addr[i] = (int*)(region0->get_address());
			}
            else {
				region1 = new mapped_region(shm, read_write);
            	std::memset(region1->get_address(), -1, region1->get_size());
            	shmem_addr[i] = (int*)(region1->get_address());
			}
            // key_t key = ftok(shmem_name,65);

    		// // shmget returns an identifier in shmid
    		// int shmid = shmget(key,1024,0666|IPC_CREAT);
			// // shmat to attach to shared memory
   			// char *str = (char*) shmat(shmid,(void*)0,0);
			// shmem_addr[i] = (int*)str;
			printf("------------- %d, region %s mapped at address %p, set to %d\n", i, shmem_name, shmem_addr[i], *(shmem_addr[i]));

        }

        int* p = shmem_addr[0];
        printf("%p\n", p);
        printf("status 2 is %d!\n", *p);

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
