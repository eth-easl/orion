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

#include "utils.h"

//void* sched_func(void* args);

class Scheduler {

	public:
		void* busy_wait_fifo(void** buffers, pthread_mutex_t** mutexes, int num_clients);
		void* busy_wait_profile(void** buffers, pthread_mutex_t** mutexes, int num_clients);
};

//void* sched_func(void* sched);
//Scheduler* sched_init();
