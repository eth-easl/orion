#ifdef DEBUG
# define DEBUG_PRINT(...) fprintf(stdout, __VA_ARGS__)
#else
# define DEBUG_PRINT(...) do {} while (0)
#endif

#include "intercept_temp.h"

using namespace std;

queue<func_record> kqueue0;
queue<func_record> kqueue1;
pthread_mutex_t mutex0;
pthread_mutex_t mutex1;
pid_t thread_ids[3]; // N threads + scheduler

queue<func_record>* kqueues[2] = {&kqueue0, &kqueue1};
pthread_mutex_t* mutexes[2] = {&mutex0, &mutex1};
int i=0;

int get_idx() {

#ifdef SYS_gettid
	pid_t tid = syscall(SYS_gettid);
#else
#error "SYS_gettid unavailable on this system"
#endif
	DEBUG_PRINT("------------------- tid is %d, %d, %d\n", tid, thread_ids[0], thread_ids[1]);
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
}


cudaError_t cudaMalloc(void** devPtr, size_t size) {

	//DEBUG_PRINT("Caught cudaMalloc! allocate region of %ld bytes\n", size);

	cudaError_t (*function)(void** devPtr, size_t size);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMalloc");
	
	cudaError_t err = (*function)(devPtr, size);
	//DEBUG_PRINT("Memory allocated at address %p, size is %ld\n", *devPtr, size);
	return err;

}


cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {

	//DEBUG_PRINT("Caught cudaMallocMANAGED! allocate region of %ld bytes\n", size);

	cudaError_t (*function)(void** devPtr, size_t size, unsigned int flags);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMallocManaged");

	cudaError_t err = (*function)(devPtr, size, flags);
	//DEBUG_PRINT("Memory allocated at address %p, size is %ld\n", *devPtr, size);
	return err;

}


cudaError_t cudaFree(void* devPtr) {

	//DEBUG_PRINT("Caught cudaFree! Free pointer that holds address %p\n", devPtr);

	cudaError_t (*function)(void* devPtr);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaFree");

	cudaError_t err; //= (*function)(devPtr);
	return err;

}



cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
	
	//DEBUG_PRINT("Caught cudaMemcpy!\n");
	
	cudaError_t (*function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMemcpy");

	cudaError_t err = (*function)(dst, src, count, kind);
	return err;

}	


cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {    

	//DEBUG_PRINT("Caught cudaMemcpyAsync! src is %p, dst is %p, size is %d\n", src, dst, count);

	cudaError_t (*function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaMemcpyAsync");

	cudaError_t err = (*function)(dst, src, count, kind, stream);
	return err;

}       



cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {


	int idx = get_idx();
	assert (idx >= 0);

	DEBUG_PRINT("Captured a cudaLaunchKernel! idx is %d, function ptr is %p, stream is %d, gridDim is %d, blockDim is %d, sharedMem is %ld\n", idx, func, stream, gridDim, blockDim, sharedMem);


	cudaError_t (*function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
	*(void **)(&function) = dlsym (RTLD_NEXT, "cudaLaunchKernel");
	cudaError_t err = cudaSuccess;

	kernel_record new_kernel_record = {func, gridDim, blockDim, args, sharedMem, stream, false, 0};

	union func_data new_func_data;
	new_func_data.krecord = new_kernel_record;
	func_record new_record = {KERNEL_RECORD, new_func_data};



	if (idx < 2) {
	
		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);
	}
	else {
		DEBUG_PRINT("------------------------ before submitting\n");
		err = (*function)(func, gridDim, blockDim, args, sharedMem, stream);
		DEBUG_PRINT("------------------------ after submitting\n");
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
	}
	
	return status;

}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {

	int idx = get_idx();
	assert (idx >= 0);
	cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
	
	DEBUG_PRINT("Caught cudnnBatchNormalizationForwardTrainingEx, handle is %p, index is %d\n", handle, idx);

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

	}

	return status;
}


cudnnStatus_t cudnnBatchNormalizationForwardInference(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon)

{

	DEBUG_PRINT("Caught cudnnBatchNormalizationForwardInference");

	int idx = get_idx();
	assert (idx >= 0);

	cudnnStatus_t (*function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon);

	*(void **)(&function) = dlsym(RTLD_NEXT, "cudnnBatchNormalizationForwardInference");
	assert(function != NULL);


	cudnnStatus_t status = (*function)(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
	return status;
}

