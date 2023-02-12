#ifdef DEBUG
# define DEBUG_PRINT(...) fprintf(stdout, __VA_ARGS__)
#else
# define DEBUG_PRINT(...) do {} while (0)
#endif

#include "intercept_temp.h"

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


using namespace std;

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
	
	DEBUG_PRINT("%d, ", i);
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

	//DEBUG_PRINT("func_names is %p, fnames0 addr is %p, %p, fnames is %p, %p\n", func_names, func_names[0], &fnames0, *(func_names[0]), fnames0);
	int idx = get_idx();
	assert (idx >= 0);
	DEBUG_PRINT("[IDX %d] Caught cudaMalloc! allocate region of %ld bytes\n", idx, size);

	cudaError_t err = cudaSuccess;
	cudaError_t (*function)(void** devPtr, size_t size);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMalloc");


	if (idx < 2) {

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

		//pthread_mutex_lock(mutexes[0]);
		kqueues[0]->pop();
		pthread_mutex_unlock(mutexes[0]);

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

		//pthread_mutex_lock(mutexes[0]);
		kqueues[0]->pop();
		pthread_mutex_unlock(mutexes[0]);


	}

	return err;

}	


cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {    

	int idx = get_idx();
	assert (idx >= 0);

	DEBUG_PRINT("[IDX: %d] Caught cudaMemcpyAsync! src is %p, dst is %p, size is %d, stream is %d\n", idx, src, dst, count, stream);

	cudaError_t (*function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMemcpyAsync");
	cudaError_t err = cudaSuccess;

	if (idx < 2) {


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

		//pthread_mutex_lock(mutexes[0]);
		kqueues[0]->pop();
		pthread_mutex_unlock(mutexes[0]);

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

	if (idx < 2) {
		block(idx);
	}

	DEBUG_PRINT("[INTERCEPTER] Captured a cudaLaunchKernel! idx is %d, function ptr is %p, stream is %d, gridDim is %d, blockDim is %d, sharedMem is %ld\n", idx, func, stream, gridDim, blockDim, sharedMem);
	print_kernel_invocation(0, gridDim, blockDim);

	cudaError_t (*function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaLaunchKernel");
	cudaError_t err = cudaSuccess;
	kernel_record new_kernel_record;
	bool wait = false;

	if (idx < 2) {

		pthread_mutex_lock(mutexes[idx]);

		// TODO: get kernel name correctly here
		char* kernel_name = func_names[idx]->at(func_indexes[idx]);
		DEBUG_PRINT("[INTERCEPTER] found a new kernel with name %s, id %d, func pointer is %p\n", kernel_name, func_indexes[idx], func);

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
		}
		else if (!strncmp(kernel_name, REDUCE_KERNEL, 44)) {
			
			// TODO: Fix this to work without waiting

			new_kernel_record = {func, gridDim, blockDim, args, sharedMem, stream, false, 0};
			wait = true;

		}
		else {

			new_kernel_record = {func, gridDim, blockDim, args, sharedMem, stream, false, 0};
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

		dim3 newgridDim(1);
		dim3 newblockDim(1);

		err = (*function)(func, gridDim, blockDim, args, sharedMem, 0);
		DEBUG_PRINT("*************** [INTERCEPTER] AFTER SUBMITTING %p *************\n", func);
		CHECK_CUDA_ERROR(err); // this checks kernel-launching errors
		
		cudaError_t err_all = cudaDeviceSynchronize(); // for debugging
		CHECK_CUDA_ERROR(err_all); // this checks (or should check) runtime-specific errors

		cudaError_t err2 = cudaGetLastError();
		CHECK_CUDA_ERROR(err2);
		//pthread_mutex_lock(mutexes[0]);
		kqueues[0]->pop();
		pthread_mutex_unlock(mutexes[0]);


	}

	// wait and run
	/* while (true) {
		pthread_mutex_lock(mutexes[0]);
		if (kqueues[0]->front().run) {
			cudaStream_t sched_stream = kqueues[0]->front().sched_stream;
			kqueues[0]->pop();
			printf("-------- run with stream %d!!!\n", sched_stream);
			pthread_mutex_unlock(mutexes[0]);
			err = (*function)(func, gridDim, blockDim, args, sharedMem, sched_stream); 
			return err;
		}
		pthread_mutex_unlock(mutexes[0]);
	} */

	return err;
}


// CUDNN ....

cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
	

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	DEBUG_PRINT("Caught cudnnConvolutionForward, CUDNN handle is %p, index is %d\n", handle, idx);

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
	}
	else {
		DEBUG_PRINT("Run Conv!!\n");
		cudnnStatus_t (*function)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) ;
		*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnConvolutionForward");
		assert(function != NULL);

		status = (*function)(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
		DEBUG_PRINT("Conv, status is %d\n", status);
		assert (status == CUDNN_STATUS_SUCCESS);

		kqueues[0]->pop();
		pthread_mutex_unlock(mutexes[0]);

	}
	
	return status;

}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
	
	DEBUG_PRINT("Caught cudnnBatchNormalizationForwardTrainingEx, handle is %p, index is %d\n", handle, idx);

	printf("%p, %d, %d, %f, %f, %p, %p, %p, %ld, %ld \n", handle, mode, bnOps, *((float*)alpha), *(float*)beta, xData, yData, zData, workSpaceSizeInBytes, reserveSpaceSizeInBytes);

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
		DEBUG_PRINT("Run BNorm!!\n");
		cudnnStatus_t (*function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);

		*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnBatchNormalizationForwardTrainingEx");
		assert(function != NULL);

		status = (*function)(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScaleData, bnBiasData, exponentialAverageFactor, resultRunningMeanData, resultRunningVarianceData, epsilon, saveMean, saveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
		assert (status == CUDNN_STATUS_SUCCESS);

		kqueues[0]->pop();
		pthread_mutex_unlock(mutexes[0]);


	}

	return status;
}


cudnnStatus_t cudnnBatchNormalizationForwardInference(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon)

{

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

	DEBUG_PRINT("Caught cudnnBatchNormalizationForwardInference, handle is %p, index is %d\n", handle, idx);

	printf("%d, %f, %f, %p, %p, %lf, %p, %p, %p, %p, %p, %p, %p\n", mode, *((float*)alpha), *((float*)beta), x, y, epsilon, estimatedMean, estimatedVariance, xDesc, yDesc, bnScaleBiasMeanVarDesc, bnScale, bnBias);

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

		//block(idx);
	}
	else {

		DEBUG_PRINT("Run BNorm Inference!!\n");

		cudnnStatus_t (*function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon);

		*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnBatchNormalizationForwardInference");
		assert(function != NULL);

		printf("%d, %f, %f, %p, %p, %lf, %p, %p, %p, %p, %p, %p, %p\n", mode, *((float*)alpha), *((float*)beta), x, y, epsilon, estimatedMean, estimatedVariance, xDesc, yDesc, bnScaleBiasMeanVarDesc, bnScale, bnBias);


		status = (*function)(handle, mode, alpha, beta, xDesc, x, xDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
		DEBUG_PRINT("BNORM, status is %d\n", status);
		assert (status == CUDNN_STATUS_SUCCESS);
		DEBUG_PRINT("return!\n");

		kqueues[0]->pop();
		pthread_mutex_unlock(mutexes[0]);



	}
		
	return status;
}


cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {

	// mock cudnn destroy TensorDescriptor
	DEBUG_PRINT("Caught a cudnnDestroyTensorDescriptor! Do nothing!\n");
	return CUDNN_STATUS_SUCCESS;
}


cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {

	DEBUG_PRINT("Caught a cudnnDestroyFilterDescriptor! Do nothing!\n");
	return CUDNN_STATUS_SUCCESS;

}


cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {

	DEBUG_PRINT("Caught a cudnnDestroyConvolutionDescriptor! Do nothing!\n");
	return CUDNN_STATUS_SUCCESS;
}
