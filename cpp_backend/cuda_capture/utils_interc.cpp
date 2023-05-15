#include "intercept_temp.h"

int get_idx() {

	#ifdef SYS_gettid
		pid_t tid = syscall(SYS_gettid);
	#else
	#error "SYS_gettid unavailable on this system"
	#endif
		//DEBUG_PRINT("-------------------my tid is %d, tids is %d, %d, %d, %d, %d \n", tid, thread_ids[0], thread_ids[1], thread_ids[2], thread_ids[3], thread_ids[4]);
		//printf("tid is %d\n", tid);
		//printf("address here is %p\n", thread_ids);

		int idx = -1;
		int clients = *num_total_clients;
		int num_tids = 2*clients+1;

		// for (int i=0; i<num_tids; i++) {
		// 	printf("tid is %d\n", thread_ids[i]);
		// }

		for (int i=0; i<num_tids; i++) {
			if (tid == thread_ids[i]) {
				idx = i%(clients+1);
				break;
			}
		}
		if (idx == -1) {
			// set threads for backward pass
			for (int i=clients+1; i<num_tids; i++) {
				if (thread_ids[i] == 0) {
					thread_ids[i] = tid;
					idx = i%(clients+1);
				}
			}
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
		//printf("size is %d\n", sz);
		if (sz==0)
			break;
	}

}