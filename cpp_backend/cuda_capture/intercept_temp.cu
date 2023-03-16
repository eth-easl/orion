#include "intercept_temp.h"

using namespace std;
using at::native::ReduceOp;
using at::_isnan;

template <typename acc_t>
struct MaxNanFunctor {
       	__device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
		return (at::_isnan(a) || a > b) ? a : b;
	}
};

template <typename acc_t>
struct MinNanFunctor {
	  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
		return (at::_isnan(a) || a < b) ? a : b;
	  }
};

template <typename T>
T* create_new_reduce_arg(void* args0) {

	T* reduce_arg = (T*)args0;
	T* new_reduce_arg = (T*)malloc(sizeof(T));
	char* dst0 = (char*)(reduce_arg->dst[0]);
	char* dst1 = (char*)(reduce_arg->dst[1]);

	*new_reduce_arg = T(
		reduce_arg->ops,
		reduce_arg->config,
		reduce_arg->input_calc,
		reduce_arg->output_calc,
		reduce_arg->src,
		dst0,
		dst1, //check this
		reduce_arg->acc_buf,
		reduce_arg->cta_buf,
		reduce_arg->semaphores,
		reduce_arg->ident,
		reduce_arg->noutputs,
		reduce_arg->base_idx
	);

	return new_reduce_arg;

}


queue<func_record> kqueue0;
queue<func_record> kqueue1;
pthread_mutex_t mutex0;
pthread_mutex_t mutex1;

vector<char*> fnames0;
vector<char*> fnames1;
volatile pid_t thread_ids[3]; // N threads + scheduler

queue<func_record>* kqueues[2] = {&kqueue0, &kqueue1};
pthread_mutex_t* mutexes[2] = {&mutex0, &mutex1};
vector<char*>* func_names[2] = {&fnames0, &fnames1};
char* model_names[2];

int func_indexes[2] = {0, 0};

void print_kernel_invocation(int i, dim3 gridDim, dim3 blockDim) {

	DEBUG_PRINT("[INTERCEPTER-CATCH]-[%d], ", i);
	if (gridDim.y == 1 && gridDim.z == 1) {
  		DEBUG_PRINT("--gridDim=%d ", gridDim.x);
	} else if (gridDim.z == 1) {
		DEBUG_PRINT("--gridDim=[%d,%d] ", gridDim.x, gridDim.y);
	} else {
		DEBUG_PRINT("--gridDim=[%d,%d,%d] ", gridDim.x, gridDim.y, gridDim.z);
	}

	if (blockDim.y == 1 && blockDim.z == 1) {
		DEBUG_PRINT("--blockDim=%d ", blockDim.x);
	} else if (blockDim.z == 1) {
		DEBUG_PRINT("--blockDim=[%d,%d] ", blockDim.x, blockDim.y);
	} else {
		DEBUG_PRINT("--blockDim=[%d,%d,%d] ", blockDim.x, blockDim.y, blockDim.z);
	}
DEBUG_PRINT("\n");
}


cudaError_t cudaMalloc(void** devPtr, size_t size) {


	int idx = get_idx();
	assert (idx >= 0);
	DEBUG_PRINT("[IDX %d] Caught cudaMalloc! allocate region of %ld bytes\n", idx, size);

	cudaError_t err = cudaSuccess;
	cudaError_t (*function)(void** devPtr, size_t size);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMalloc");


	if (idx < 2) {

		// wait for all kernels or memory operations to finish
		block(idx,  mutexes, kqueues);

		malloc_record new_malloc_record = {devPtr, size};
		union func_data new_func_data;
		new_func_data.malrecord = new_malloc_record;
		func_record new_record = {MALLOC_RECORD, new_func_data};


		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		// wait for mem to be allocated
		block(idx,  mutexes, kqueues);
		DEBUG_PRINT("[IDX %d] Exit malloc!\n", idx);
	}

	else {

		err = (*function)(devPtr, size);
		CHECK_CUDA_ERROR(err);
		cudaError_t err_all = cudaDeviceSynchronize();
		CHECK_CUDA_ERROR(err_all);
		DEBUG_PRINT("[IDX %d] Malloc Done!!\n", idx);

	}

	return err;

}


cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {

	//DEBUG_PRINT("Caught cudaMallocMANAGED! allocate region of %ld bytes\n", size);

	cudaError_t (*function)(void** devPtr, size_t size, unsigned int flags);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMallocManaged");

	cudaError_t err = (*function)(devPtr, size, flags);
	CHECK_CUDA_ERROR(err);
	//DEBUG_PRINT("Memory allocated at address %p, size is %ld\n", *devPtr, size);
	return err;

}


cudaError_t cudaFree(void* devPtr) {

	DEBUG_PRINT("*********************************** Caught cudaFree! Free pointer that holds address %p\n", devPtr);

	cudaError_t (*function)(void* devPtr);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaFree");

	cudaError_t err = cudaSuccess; //= (*function)(devPtr);
	CHECK_CUDA_ERROR(err);
	return err;

}



cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {

	int idx = get_idx();
	assert (idx >= 0);
	DEBUG_PRINT("[IDX: %d], Caught cudaMemcpy!\n", idx);

	cudaError_t (*function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMemcpy");
	cudaError_t err = cudaSuccess;

	if (idx < 2) {

		// wait for all kernels or memory operations to finish
		block(idx,  mutexes, kqueues);
		memcpy_record new_memcpy_record = {dst, src, count, kind, 0, false};

		union func_data new_func_data;
		new_func_data.mrecord = new_memcpy_record;
		func_record new_record = {MEMCPY_RECORD, new_func_data};

		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		// wait for memcpy to finish
		block(idx,  mutexes, kqueues);
	}

	else {

		err = (*function)(dst, src, count, kind);
		CHECK_CUDA_ERROR(err);
		cudaError_t err_all = cudaDeviceSynchronize();
		CHECK_CUDA_ERROR(err_all);

	}

	return err;

}


cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {

	int idx = get_idx();
	assert (idx >= 0);

	//DEBUG_PRINT("[IDX: %d] Caught cudaMemcpyAsync! src is %p, dst is %p, size is %d, stream is %d\n", idx, src, dst, count, stream);

	cudaError_t (*function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMemcpyAsync");
	cudaError_t err = cudaSuccess;

	if (idx < 2) {

		// wait for all kernels or memory operations to finish
		block(idx,  mutexes, kqueues);

		memcpy_record new_memcpy_record = {dst, src, count, kind, stream, true};

		union func_data new_func_data;
		new_func_data.mrecord = new_memcpy_record;
		func_record new_record = {MEMCPY_RECORD, new_func_data};

		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		// although async, wait for debugging purposes
		block(idx,  mutexes, kqueues);
	}

	else {

		err = (*function)(dst, src, count, kind, 0); // TODO: not sure about which stream to use here
		CHECK_CUDA_ERROR(err);
		cudaError_t err_all = cudaDeviceSynchronize();
		CHECK_CUDA_ERROR(err_all);
	}

	return err;

}


cudaError_t cudaMemset(void* devPtr, int  value, size_t count ) {

	printf("----------- Caught CUDA_MEMSET!!!!!!!!!!!!!\n");
	cudaError_t (*function)(void* devPtr, int value, size_t count);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMemset");

	cudaError_t err = (*function)(devPtr, value, count);
	CHECK_CUDA_ERROR(err);
	return err;
}


cudaError_t cudaMemsetAsync ( void* devPtr, int  value, size_t count, cudaStream_t stream) {

	printf("----------- Caught CUDA_MEMSET_ASYNC!!!!!!!!!!!!!\n");
	cudaError_t (*function)(void* devPtr, int value, size_t count, cudaStream_t stream);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMemsetAsync");

	cudaError_t err = (*function)(devPtr, value, count, stream);
	CHECK_CUDA_ERROR(err);
	return err;


}


cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {


	int idx = get_idx();
	assert (idx >= 0);

	// TODO: remove this
	if (idx < 2)
		block(idx,  mutexes, kqueues);

	if (idx < 2)
		DEBUG_PRINT("------------------------- IDX %d, model name is %s\n", idx, model_names[idx]);

	//DEBUG_PRINT("[INTERCEPTER-CATCH] Captured a cudaLaunchKernel! idx is %d, function ptr is %p, stream is %d, gridDim is %d, blockDim is %d, sharedMem is %ld\n", idx, func, stream, gridDim, blockDim, sharedMem);
	print_kernel_invocation(func_indexes[idx], gridDim, blockDim);

	cudaError_t (*function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaLaunchKernel");
	cudaError_t err = cudaSuccess;
	kernel_record new_kernel_record;
	bool wait = false;

	if (idx < 2) {

		pthread_mutex_lock(mutexes[idx]);

		// TODO: get kernel name correctly here
		char* kernel_name = func_names[idx]->at(func_indexes[idx]);
		DEBUG_PRINT("[INTERCEPTER] found a new kernel id %d, name is %s, func pointer is %p\n", func_indexes[idx], kernel_name, func);

		if (!strncmp(kernel_name, VECTORIZED_ELEMENTWISE_KERNEL, 41)) {

			// NOTE: WE EXPECT AN ADDITIONAL ARGUMENT WITH THE NUMBER OF INPUT/OUTPUT TENSORS
			// TODO: How to get this during runtime?
			void** new_args = (void**)malloc(4*sizeof(void*));
			// first arg: int

 			int* first_arg = (int*)malloc(sizeof(int));
			new_args[0] = first_arg;
			*first_arg = *((int*)(args[0]));

			new_args[1] = args[1];

			int data_size = *((int*)args[2]);
			int* data_size_ptr = (int*)malloc(sizeof(int));
			*data_size_ptr = data_size;
			printf("******************* IDX IS %d, DATA SIZE IS %d\n", func_indexes[idx], data_size);
			Array<char*, 10>* data_ptr = (Array<char*, 10>*)malloc(sizeof(Array<char*, 10>));
			for (int i=0; i<data_size; i++) {
				data_ptr->data[i] = ((Array<char*, 10>*)args[3])->data[i];
				printf("POINTER AT INDEX %d IS %p\n", i, data_ptr->data[i]);
			}

			new_args[2] = data_size_ptr;
			new_args[3] = data_ptr;


			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};

		}
		else if (!strncmp(kernel_name, CUB_DEVICE_REDUCE_SINGLE_TILE_KERNEL, 54)) {

			void** new_args = (void**)malloc(5*sizeof(void*));
			new_args[0] = args[0];
			new_args[1] = args[1];

			new_args[2] = (int*)malloc(sizeof(int));
			*((int*)new_args[2]) = *((int*)(args[2]));

			new_args[3] = args[3];

			new_args[4] = (int*)malloc(sizeof(int));
			*((int*)new_args[4]) = *((int*)(args[4]));

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			wait = true;

		}
		else if (!strncmp(kernel_name, CUB_DEVICE_COMPACT_INIT_KERNEL, 49)) {

			void** new_args = (void**)malloc(3*sizeof(void*));

			new_args[0] = args[0];

			new_args[1] = (int*)malloc(sizeof(int));
			*((int*)new_args[1]) = *((int*)(args[1]));

			new_args[2] = args[2];

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			wait = true;
		}
		else if (!strncmp(kernel_name, CUB_DEVICE_SELECT_SWEEP_KERNEL, 49)) {

			void** new_args = (void**)malloc(9*sizeof(void*));
			for (int i=0; i<7; i++)
				new_args[i] = args[i];

			new_args[7] = (int*)malloc(sizeof(int));
			*((int*)new_args[7]) = *((int*)(args[7]));

			new_args[8] = (int*)malloc(sizeof(int));
			*((int*)new_args[8]) = *((int*)(args[8]));

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			wait = true;

		}
		else if (!strncmp(kernel_name, INDEX_ELEMENTWISE_KERNEL, 41)) {

			void** new_args = (void**)malloc(2*sizeof(void*));

			new_args[0] = (int*)malloc(sizeof(int));
			*((int*)new_args[0]) = *((int*)(args[0]));

			new_args[1] = args[1];

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			wait = true; // leave this for now
		}
		else if (!strncmp(kernel_name, UNROLLED_ELEMENTWISE_KERNEL, 44)) {

			// NOTE: WE EXPECT AN ADDITIONAL ARGUMENT WITH THE NUMBER OF INPUT/OUTPUT TENSORS
			// TODO: How to get this during runtime?

			void** new_args = (void**)malloc(8*sizeof(void*));
			new_args[0] = (int*)malloc(sizeof(int));
			*((int*)new_args[0]) = *((int*)(args[0]));

			new_args[1] = args[1];
			new_args[4] = args[4];
			new_args[5] = args[5];
			new_args[6] = args[6];
			new_args[7] = args[7];

			int data_size = *((int*)args[2]);
			int* data_size_ptr = (int*)malloc(sizeof(int));
			*data_size_ptr = data_size;
			printf("******************* IDX IS %d, DATA SIZE IS %d\n", func_indexes[idx], data_size);
			Array<char*, 10>* data_ptr = (Array<char*, 10>*)malloc(sizeof(Array<char*, 10>));
			for (int i=0; i<data_size; i++) {
				data_ptr->data[i] = ((Array<char*, 10>*)args[3])->data[i];
				printf("POINTER AT INDEX %d IS %p\n", i, data_ptr->data[i]);
			}

			new_args[2] = data_size_ptr;
			new_args[3] = data_ptr;

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			// TODO: why BERT has problem here?
			wait = true;
		}
		else if (!strncmp(kernel_name, REDUCE_KERNEL, 44)) {

			void** new_args = (void**)malloc(sizeof(void*));
			if (!strcmp(model_names[idx], MOBILENET) && func_indexes[idx] == 149) {

				using arg_type = at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>;
				arg_type* new_reduce_arg = create_new_reduce_arg<arg_type>(args[0]);
				new_args[0] = new_reduce_arg;
			}
			else if (!strcmp(model_names[idx], GNMT) && func_indexes[idx] == 35) {
				using arg_type = at::native::ReduceOp<float, at::native::NormTwoOps<float, float>, unsigned int, float, 4>;
				arg_type* new_reduce_arg = create_new_reduce_arg<arg_type>(args[0]);
				new_args[0] = new_reduce_arg;

			}
			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
		}
		else if (!strncmp(kernel_name, MAX_POOL_FORWARD_NCHW, 61)) {

			void** new_args = (void**)malloc(17*sizeof(void*));

			new_args[0]  = (int*)malloc(sizeof(int));
			*((int*)new_args[0]) = *((int*)(args[0]));
			for (int i=2; i<15; i++) {
				new_args[i] = (int*)malloc(sizeof(int));
				*((int*)new_args[i]) = *((int*)(args[i]));
			}
			new_args[1] = args[1];
			new_args[15] = args[15];
			new_args[16] = args[16];

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};

			// TODO: check why invalid memory accesses here (for both reads and writes)
			wait = true;
		}
		else if (!strncmp(kernel_name, ELEMENTWISE_KERNEL_WITH_INDEX, 57)) {

			void** new_args = (void**)malloc(3*sizeof(void*));
			new_args[0]  = (int*)malloc(sizeof(int));
			*(((int*)new_args[0])) = *((int*)(args[0]));

			new_args[1] = args[1];
			new_args[2] = args[2];

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			// used in bert, only once so just wait
			wait = true;
		}
		else if (!strncmp(kernel_name, INDEX_SELECT_LARGE_INDEX, 61)) {

			void** new_args = (void**)malloc(8*sizeof(void*));
			new_args[0] = args[0];
			new_args[1] = args[1];
			new_args[2] = args[2];

			for (int i=3; i<5; i++) {
				new_args[i] = (int*)malloc(sizeof(int));
				*(((int*)new_args[i])) = *((int*)(args[i]));
			}

			for (int i=5; i<7; i++) {
				new_args[i] = (unsigned int*)malloc(sizeof(unsigned int));
				*(((unsigned int*)new_args[i])) = *((unsigned int*)(args[i]));
			}

			new_args[7] = (int64_t*)malloc(sizeof(int64_t));
			*(((int64_t*)new_args[7])) = *((int64_t*)(args[7]));

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			// invalid memory acces - why?
			wait = true;
		}
		else if (!strncmp(kernel_name, ELEMENTWISE_KERNEL, 35)) {

			void** new_args = (void**)malloc(8*sizeof(void*));

			new_args[0] = (int*)malloc(sizeof(int));
			*(((int*)new_args[0])) = *((int*)(args[0]));

			new_args[1] = args[1];
			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			// TODO: VERY IMPORTANT - invalid memory access - why?
			wait = true;
		}
		else if (!strncmp(kernel_name, SOFTMAX_WARP_FORWARD, 48)) {

			void** new_args = (void**)malloc(8*sizeof(void*));
			new_args[0] = args[0];
			new_args[1] = args[1];
			new_args[5] = args[5];

			for (int i=2; i<5; i++) {
				new_args[i] = (int*)malloc(sizeof(int));
				*(((int*)new_args[i])) = *((int*)(args[i]));
			}

			new_args[6] = (int*)malloc(sizeof(int));
			*(((int*)new_args[6])) = *((int*)(args[6]));

			new_args[7] = (bool*)malloc(sizeof(bool));
			*(((bool*)new_args[7])) = *((bool*)(args[7]));

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			// TODO: FIXME!
			wait = true;
		}
		else if (!strncmp(kernel_name, VECTORIZED_LAYER_NORM_KERNEL, 68)) {

			void** new_args = (void**)malloc(8*sizeof(void*));

			new_args[0] = (int*)malloc(sizeof(int));
			*(((int*)new_args[0])) = *((int*)(args[0]));

			new_args[1] = (float*)malloc(sizeof(float));
			*(((float*)new_args[1])) = *((int*)(args[1]));

			for (int i=2; i<8; i++)
				new_args[i] = args[i];

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			// TODO: FIXME!!!!!!
			wait = true;
		}
		else if (!strncmp(kernel_name, TRIU_TRIL_KERNEL, 33)) {

			void** new_args = (void**)malloc(4*sizeof(void*));
			new_args[0] = args[0];
			new_args[1] = args[1];

			for (int i=2; i<4; i++) {
				new_args[i] = (int64_t*)malloc(sizeof(int64_t));
				*(((int64_t*)new_args[i])) = *((int64_t*)(args[i]));
			}
			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			wait = true;
		}
		else if (!strncmp(kernel_name, CAT_ARRAY_BATCHED_COPY, 59)) {

			void** new_args = (void**)malloc(5*sizeof(void*));
			for (int i=0; i<3; i++)
				new_args[i] = args[i];

			new_args[3] = (int*)malloc(sizeof(int));
			*(((int*)new_args[3])) = *((int*)(args[3]));

			new_args[4] = (unsigned int*)malloc(sizeof(unsigned int));
			*(((unsigned int*)new_args[4])) = *((unsigned int*)(args[4]));

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			// TODO: FIXME
			wait = true;
		}
		else if (!strncmp(kernel_name, UPSAMPLE_BILINEAR2D_OUT_FRAME, 69)) {

			void** new_args = (void**)malloc(6*sizeof(void*));

			new_args[0] = (int*)malloc(sizeof(int));
			*(((int*)new_args[0])) = *((int*)(args[0]));

			new_args[1] = (float*)malloc(sizeof(float));
			*(((float*)new_args[1])) = *((float*)(args[1]));

			new_args[2] = (float*)malloc(sizeof(float));
			*(((float*)new_args[2])) = *((float*)(args[2]));

			new_args[3] = (bool*)malloc(sizeof(bool));
			*(((bool*)new_args[3])) = *((bool*)(args[3]));

			new_args[4] = args[4];
			new_args[5] = args[5];

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};

			// TODO: FIXME
			wait = true;
		}
		else if (!strncmp(kernel_name, UPSAMPLE_NEAREST2D_NHWC_OUT_FRAME, 73)) {

			void** new_args = (void**)malloc(10*sizeof(void*));

			new_args[0] = args[0];
			new_args[1] = args[1];

			for (int i=2; i<7; i++){
				new_args[i] = (size_t*)malloc(sizeof(size_t));
				*(((size_t*)new_args[i])) = *((size_t*)(args[i]));
			}

			new_args[9] = (size_t*)malloc(sizeof(size_t));
			*(((size_t*)new_args[9])) = *((size_t*)(args[9]));

			for (int i=7; i<9; i++){
				new_args[i] = (float*)malloc(sizeof(float));
				*(((float*)new_args[i])) = *((float*)(args[i]));
			}

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			// TODO: FIXME
			wait = true;
		}
		else if (!strncmp(kernel_name, CUB_DEVICE_REDUCE_KERNEL, 44)) {

			void** new_args = (void**)malloc(5*sizeof(void*));
			new_args[0] = args[0];
			new_args[1] = args[1];
			new_args[3] = args[3];
			new_args[4] = args[4];

			new_args[2] = (int*)malloc(sizeof(int));
			*(((int*)new_args[2])) = *((int*)(args[2]));

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			wait = true;
		}
		else if (!strncmp(kernel_name, CUB_DEVICE_SCAN_INIT_KERNEL, 46)) {

			void** new_args = (void**)malloc(2*sizeof(void*));
			new_args[0] = args[0];

			new_args[1] = (size_t*)malloc(sizeof(size_t));
			*(((size_t*)new_args[1])) = *((size_t*)(args[1]));

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
			wait = true;
		}
		else if (!strncmp(kernel_name, CUB_DEVICE_SCAN_KERNEL, 42)) {


			new_kernel_record = {func, gridDim, blockDim, args, sharedMem, stream, false, 0};
			wait = true;

		}
		else {

			new_kernel_record = {func, gridDim, blockDim, args, sharedMem, stream, false, 0};
			//wait = true;
		}



		union func_data new_func_data;
		new_func_data.krecord = new_kernel_record;
		func_record new_record = {KERNEL_RECORD, new_func_data};

		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		func_indexes[idx] += 1;

		//if (wait)
		block(idx,  mutexes, kqueues);

	}
	else {
		DEBUG_PRINT("[INTERCEPTER] about to submit %p\n", func);

		err = (*function)(func, gridDim, blockDim, args, sharedMem, 0);
		DEBUG_PRINT("*************** [INTERCEPTER] AFTER SUBMITTING %p *************\n", func);
		CHECK_CUDA_ERROR(err); // this checks kernel-launching errors

		cudaError_t err_all = cudaDeviceSynchronize(); // for debugging
		CHECK_CUDA_ERROR(err_all); // this checks (or should check) runtime-specific errors

		cudaError_t err2 = cudaGetLastError();
		CHECK_CUDA_ERROR(err2);



	}
	return err;
}


// CUDNN ....


// CUBLAS ....
