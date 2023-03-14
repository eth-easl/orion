#ifdef DEBUG
# define DEBUG_PRINT(...) fprintf(stdout, __VA_ARGS__)
#else
# define DEBUG_PRINT(...) do {} while (0)
#endif

#include "intercept_temp.h"

using namespace std;
using at::native::ReduceOp;
using at::_isnan;


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
		           const int line)
{
	if (err != cudaSuccess)
	{
		printf("CUDA Runtime Error at: %s:%d\n", file, line);
		printf("Error %d, %s\n", err, cudaGetErrorString(err));
	}
	assert (err == cudaSuccess);
}


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
int i=0;

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

void block(int idx) {
	
	while (1) {
		pthread_mutex_lock(mutexes[idx]);
		volatile int sz = kqueues[idx]->size(); // wait. TODO: is this needed?
		pthread_mutex_unlock(mutexes[idx]);
		if (sz==0)
			break;
	}

}

cudaError_t cudaMalloc(void** devPtr, size_t size) {


	int idx = get_idx();
	assert (idx >= 0);
	//DEBUG_PRINT("[IDX %d] Caught cudaMalloc! allocate region of %ld bytes\n", idx, size);

	cudaError_t err = cudaSuccess;
	cudaError_t (*function)(void** devPtr, size_t size);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMalloc");
	

	if (idx < 2) {

		// wait for all kernels or memory operations to finish
		block(idx);

		malloc_record new_malloc_record = {devPtr, size};
		union func_data new_func_data;
		new_func_data.malrecord = new_malloc_record;
		func_record new_record = {MALLOC_RECORD, new_func_data};


		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		// wait for mem to be allocated
		block(idx);


	}

	else {

		err = (*function)(devPtr, size);
		CHECK_CUDA_ERROR(err);
		cudaError_t err_all = cudaDeviceSynchronize();
		CHECK_CUDA_ERROR(err_all);

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
		block(idx);

		memcpy_record new_memcpy_record = {dst, src, count, kind, 0, false};

		union func_data new_func_data;
		new_func_data.mrecord = new_memcpy_record;
		func_record new_record = {MEMCPY_RECORD, new_func_data};

		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		// wait for memcpy to finish
		block(idx);
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
		block(idx);

		memcpy_record new_memcpy_record = {dst, src, count, kind, stream, true};

		union func_data new_func_data;
		new_func_data.mrecord = new_memcpy_record;
		func_record new_record = {MEMCPY_RECORD, new_func_data};

		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		// although async, wait for debugging purposes
		block(idx);
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

	//if (idx < 2)
	//	block(idx);

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
			

		}
		else if (!strncmp(kernel_name, CUB_DEVICE_COMPACT_INIT_KERNEL, 49)) {
			
			void** new_args = (void**)malloc(3*sizeof(void*));
			
			new_args[0] = args[0];
			
			new_args[1] = (int*)malloc(sizeof(int));
			*((int*)new_args[1]) = *((int*)(args[1]));

			new_args[2] = args[2];

			new_kernel_record = {func, gridDim, blockDim, new_args, sharedMem, stream, false, 0};
		
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
		
		if (wait)
			block(idx);

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

cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
	

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	DEBUG_PRINT("[INTERCEPTER-CATCH]-[%d] Caught cudnnConvolutionForward, CUDNN handle is %p, index is %d\n", func_indexes[idx], handle, idx);

	// create record
	cudnnConvolutionForward_record new_conv_record = {
		handle,
		alpha,
		xDesc,
		x,
		wDesc,
		w,
		convDesc,
		algo,
		workSpace,
		workSpaceSizeInBytes,
		beta,
		yDesc,
		y
	};
	union func_data new_func_data;
	new_func_data.cudnnConvRecord = new_conv_record;
	func_record new_record = {CUDNN_CONV_RECORD, new_func_data};
	
	

	// push or run
	if (idx < 2) {
		 pthread_mutex_lock(mutexes[idx]);
		 kqueues[idx]->push(new_record);
		 pthread_mutex_unlock(mutexes[idx]);

		 func_indexes[idx] += 1;
	}
	else {
		cudnnStatus_t (*function)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) ;
		*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnConvolutionForward");
		assert(function != NULL);

		status = (*function)(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
		assert (status == CUDNN_STATUS_SUCCESS);

	}
	
	return status;

}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
	
	DEBUG_PRINT("[INTERCEPTER] Caught cudnnBatchNormalizationForwardTrainingEx, handle is %p, index is %d\n", handle, idx);


	// create record
	cudnnBatchNormalizationForwardTrainingEx_record new_bn_record = {
		handle,
		mode,
		bnOps,
		alpha,
		beta,
		xDesc,
		xData,
		zDesc,
		zData,
		yDesc,
		yData,
		bnScaleBiasMeanVarDesc,
		bnScaleData,
		bnBiasData,
		exponentialAverageFactor,
		resultRunningMeanData,
		resultRunningVarianceData,
		epsilon,
		saveMean,
		saveInvVariance,
		activationDesc,
		workspace,
		workSpaceSizeInBytes,
		reserveSpace,
		reserveSpaceSizeInBytes

	};
	union func_data new_func_data;
	new_func_data.cudnnBNormRecord = new_bn_record;
	func_record new_record = {CUDNN_BNORM_RECORD, new_func_data};

	// push or run

	if (idx < 2) { 
		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

	}
	else {
		cudnnStatus_t (*function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);

		*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnBatchNormalizationForwardTrainingEx");
		assert(function != NULL);

		status = (*function)(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScaleData, bnBiasData, exponentialAverageFactor, resultRunningMeanData, resultRunningVarianceData, epsilon, saveMean, saveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
		assert (status == CUDNN_STATUS_SUCCESS);

	}

	return status;
}


cudnnStatus_t cudnnBatchNormalizationForwardInference(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon)

{

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	DEBUG_PRINT("[INTERCEPTER-CATCH]-[%d] Caught cudnnBatchNormalizationForwardInference, handle is %p, index is %d\n", func_indexes[idx], handle, idx);



	// create record
	cudnnBatchNormalizationForwardInference_record bn_record = {
		handle,
		mode,
		alpha,
		beta,
		xDesc,
		x,
		yDesc,
		y,
		bnScaleBiasMeanVarDesc,
		bnScale,
		bnBias,
		estimatedMean,
		estimatedVariance,
		epsilon
	};

	union func_data new_func_data;
	new_func_data.cudnnBNormInfRecord = bn_record;
	func_record new_record = {CUDNN_BNORM_INF_RECORD, new_func_data};

	if (idx < 2) {
		
		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);
		
		func_indexes[idx] += 1;

	}
	else {

		cudnnStatus_t (*function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon);

		*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnBatchNormalizationForwardInference");
		assert(function != NULL);

		status = (*function)(handle, mode, alpha, beta, xDesc, x, xDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
		assert (status == CUDNN_STATUS_SUCCESS);

	}
		
	return status;
}


cudnnStatus_t cudnnRNNForwardInference(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes)  {

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;


	DEBUG_PRINT("[INTERCEPTER-CATCH]-[%d] Caught cudnnRNNForwardInference, handle is %p, index is %d\n", func_indexes[idx], handle, idx);
	printf("------------------------------------------------- IDX [%d], CX IS %p, CY IS %p\n", idx, cx, cy);
	
	if (idx < 2) {

		cudnnTensorDescriptor_t* xDesc_new = (cudnnTensorDescriptor_t*)malloc(sizeof(cudnnTensorDescriptor_t));
	        //cudnnStatus_t s = cudnnCreateTensorDescriptor(xDesc_new);

		*xDesc_new = *xDesc;
		printf("%p, %p, %p, %p\n", xDesc, *xDesc, xDesc_new, *(xDesc_new));
		//memcpy(xDesc_new, xDesc, sizeof(cudnnTensorDescriptor_t));

		cudnnTensorDescriptor_t* yDesc_new = (cudnnTensorDescriptor_t*)malloc(sizeof(cudnnTensorDescriptor_t));
		*yDesc_new = *yDesc;


		cudnnRNNForwardInference_record rnn_record = {
			handle,
			rnnDesc,
			seqLength,
			xDesc_new,
			x,
			hxDesc,
			hx,
			cxDesc,
			cx,
			wDesc,
			w,
			yDesc_new,
			y,
			hyDesc,
			hy,
			cyDesc,
			cy,
			workspace,
			workSpaceSizeInBytes
		};

		union func_data new_func_data;
		new_func_data.cudnnRnnInfRecord = rnn_record;
		func_record new_record = {CUDNN_RNN_INF_RECORD, new_func_data};

		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		func_indexes[idx] += 1;
	}
	else {
		cudnnStatus_t (*function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes);

		
		
		*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnRNNForwardInference");
		assert(function != NULL);

		status = (*function)(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes);

		
		printf("------------------------- cudnn status is %d\n", status);
		// TODO: not sure why this complains here in just one call!
		//assert (status == CUDNN_STATUS_SUCCESS);	

		cudaError_t err_all = cudaDeviceSynchronize(); // for debugging
		CHECK_CUDA_ERROR(err_all); 
	}

	return status;

}

cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
	
	//DEBUG_PRINT("Caught a cudnnDestroyRNNDescriptor! Do nothing!\n");	
	return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {

	// mock cudnn destroy TensorDescriptor
	//DEBUG_PRINT("Caught a cudnnDestroyTensorDescriptor! Do nothing!\n");
	return CUDNN_STATUS_SUCCESS;
}


cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {

	//DEBUG_PRINT("Caught a cudnnDestroyFilterDescriptor! Do nothing!\n");
	return CUDNN_STATUS_SUCCESS;

}


cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {

	//DEBUG_PRINT("Caught a cudnnDestroyConvolutionDescriptor! Do nothing!\n");
	return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
	//DEBUG_PRINT("Caught a cudnnDestroyDropoutDescriptor! Do nothing!\n");
	return CUDNN_STATUS_SUCCESS;

}


// CUBLAS ....

cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {

	int idx = get_idx();
	assert (idx >= 0);
	cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
	
	cublasSgemm_record blassgemm_record = {
		handle,
		transa,
		transb,
		m,
		n,
		k,
		alpha,
		A,
		lda,
		B,
		ldb,
		beta,
		C,
		ldc
	};

	union func_data new_func_data;
	new_func_data.cublasSgemmRecord = blassgemm_record;
	func_record new_record = {CUBLAS_SGEMM_RECORD, new_func_data};

	printf("Intercepter func is %p\n", cublasSgemm);

	if (idx < 2) {

		DEBUG_PRINT("[INTERCEPTER-CATCH]-[%d] Caught cublasSgemm, handle is %p, index %d, m is %d, n is %d, k is %d\n", func_indexes[idx], handle, idx, m, n, k);

		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		func_indexes[idx] += 1;
	}
	else {

		cublasStatus_t (*function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);

		*(void **)(&function) = dlsym(RTLD_NEXT, "cublasSgemm_v2");
		assert(function != NULL);
		
		status = (*function)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		assert (status == CUBLAS_STATUS_SUCCESS);
		DEBUG_PRINT("CUBLAS status is %d\n", status);

	}
	
	return status;

}


cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount) {

	int idx = get_idx();
	assert (idx >= 0);
	cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

	cublasSgemmStridedBatched_record record = {
		handle,
		transa,
		transb,
		m,
		n,
		k,
		alpha,
		A,
		lda,
		strideA,
		B,
		ldb,
		strideB,
		beta,
		C,
		ldc,
		strideC,
		batchCount
	};

	union func_data new_func_data;
	new_func_data.cublasSgemmStridedRecord = record;
	func_record new_record = {CUBLAS_SGEMM_STRIDED_RECORD, new_func_data};

	if (idx < 2) {

		DEBUG_PRINT("[INTERCEPTER-CATCH]-[%d] Caught cublasSgemmStridedBatched, handle is %p\n", func_indexes[idx], handle);
	
		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		func_indexes[idx] += 1;
	
	}
	else {

		cublasStatus_t (*function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount);

		*(void **)(&function) = dlsym(RTLD_NEXT, "cublasSgemmStridedBatched");
		assert(function != NULL);

		status = (*function)(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
		assert (status == CUBLAS_STATUS_SUCCESS);
		DEBUG_PRINT("CUBLAS status is %d\n", status);
	}

	return status;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {

	DEBUG_PRINT("Caught a cublasDestroy! Do nothing!\n");
	return CUBLAS_STATUS_SUCCESS;
}
