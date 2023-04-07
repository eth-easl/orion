#include "intercept_temp.h"

int get_idx() {

	#ifdef SYS_gettid
		pid_t tid = syscall(SYS_gettid);
	#else
	#error "SYS_gettid unavailable on this system"
	#endif
		//DEBUG_PRINT("------------------- tid is %d, %d, %d, %d\n", tid, thread_ids[0], thread_ids[1], thread_ids[2]);
		if (tid == thread_ids[0])
			return 0;
		else if (tid == thread_ids[1])
			return 1;
		else if (tid == thread_ids[2])
			return 2;
		else
			return -1;
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