#include "intercept_temp.h"

int get_idx() {

	return 0;

	// #ifdef SYS_gettid
	// 	pid_t tid = syscall(SYS_gettid);
	// #else
	// #error "SYS_gettid unavailable on this system"
	// #endif
	// 	//DEBUG_PRINT("-------------------my tid is %d, tids is %d, %d, %d, %d, %d \n", tid, thread_ids[0], thread_ids[1], thread_ids[2], thread_ids[3], thread_ids[4]);
	// 	//printf("tid is %d\n", tid);
	// 	if ((tid == thread_ids[0]) || (tid == thread_ids[3]))
	// 		return 0;
	// 	else if ((tid == thread_ids[1]) || (tid == thread_ids[4]))
	// 		return 1;
	// 	else if (tid == thread_ids[2])
	// 		return 2;
	// 	else if (thread_ids[3] == 0) {
	// 		thread_ids[3] = tid;
	// 		return 0;
	// 	}
	// 	else if (thread_ids[4] == 0) {
	// 		thread_ids[4] = tid;
	// 		return 1;
	// 	}
	// 	else
	// 		return -1;
}

void block(int idx, pthread_mutex_t** mutexes, queue<func_record>** kqueues) {

	while (1) {
		pthread_mutex_lock(mutexes[idx]);
		volatile int sz = kqueues[idx]->size(); // wait. TODO: is this needed?
		pthread_mutex_unlock(mutexes[idx]);
		if (sz==0)
			break;
	}

}