#include <stdio.h>
//#include <cublas.h>
#include <dlfcn.h>
#include <queue>
#include <vector>
#include <pthread.h>
#include <syscall.h>
#include <pwd.h>
#include <iostream>
#include <string.h>
#include <fstream>

#include "utils_sched.h"

//void* sched_func(void* args);

class Scheduler {

	public:
		void fifo_prep(void** qbuffers, int num_clients);
		void profile_prep(void** qbuffers, int num_clients);
		void profile_reset(int num_clients);
		void* busy_wait_fifo(int num_clients);
		void* busy_wait_profile(int num_clients, int iter);
		void* busy_wait_single_client(int client_id);
};

//void* sched_func(void* sched);
//Scheduler* sched_init();
