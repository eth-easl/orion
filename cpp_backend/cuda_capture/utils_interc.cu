#include "intercept_temp.h"

int get_idx() {

	#ifdef SYS_gettid
		pid_t tid = syscall(SYS_gettid);
	#else
	#error "SYS_gettid unavailable on this system"
	#endif
		//DEBUG_PRINT("-------------------my tid is %d, tids is %d, %d, %d, %d, %d \n", tid, thread_ids[0], thread_ids[1], thread_ids[2], thread_ids[3], thread_ids[4]);
		//printf("tid is %d\n", tid);
		int idx = -1;
		if ((tid == thread_ids[0]) || (tid == thread_ids[3]))
			idx = 0;
		else if ((tid == thread_ids[1]) || (tid == thread_ids[4]))
			idx = 1;
		else if (tid == thread_ids[2])
			idx = 2;
		else if (thread_ids[3] == 0) {
			thread_ids[3] = tid;
			idx = 0;
		}
		else if (thread_ids[4] == 0) {
			thread_ids[4] = tid;
			idx = 1;
		}
		if (idx > -1) {
			cpu_set_t  mask;
			CPU_ZERO(&mask);
			CPU_SET(idx+4, &mask);
			int result = sched_setaffinity(0, sizeof(mask), &mask);
			assert (result==0);
		}
		return idx;
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