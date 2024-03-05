/* Intercepts and overwrites CUDA calls */


#include "intercept_temp.h"

using namespace std;

volatile pid_t* thread_ids; // 2*N threads + scheduler

queue<func_record>** kqueues;
pthread_mutex_t** mutexes;

int* func_indexes;
int* num_total_clients;

volatile bool** client_request_status;
volatile bool* client_stop;
volatile bool* client_stop_ack;
volatile bool* affinity_set;


cudaError_t (*kernel_func)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) = NULL;
cudaError_t (*memcpy_func)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) = NULL;
cudaError_t (*memcpy_async_func)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) = NULL;
cudaError_t (*malloc_func)(void** devPtr, size_t size) = NULL;
cudaError_t (*free_func)(void* devPtr) = NULL;
cudaError_t (*memset_func)(void* devPtr, int  value, size_t count) = NULL;
cudaError_t (*memset_async_func)(void* devPtr, int  value, size_t count, cudaStream_t stream) = NULL;

cudnnStatus_t (*cudnn_conv_func)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) = NULL;
cudnnStatus_t (*cudnn_bnorm_func)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) = NULL;
cudnnStatus_t (*cudnn_bnorm_infer_func)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon) = NULL;
cudnnStatus_t (*cudnn_rnn_func)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes) = NULL;
cudnnStatus_t (*cudnn_rnn_train_func)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) = NULL;
cublasStatus_t (*cublas_sgemm_func)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) = NULL;
cublasStatus_t (*cublas_sgemm_strided_func)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount) = NULL;

cudnnStatus_t (*cudnn_bnorm_bw_func)(
	cudnnHandle_t handle,
	cudnnBatchNormMode_t mode,
	cudnnBatchNormOps_t bnOps,
	const void *alphaDataDiff,
    const void *betaDataDiff,
    const void *alphaParamDiff,
    const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t yDesc,
    const void *yData,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dyData,
    const cudnnTensorDescriptor_t dzDesc,
    void *dzData,
    const cudnnTensorDescriptor_t dxDesc,
    void *dxData,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScaleData,
    const void *bnBiasData,
    void *dBnScaleData,
    void *dBnBiasData,
    double epsilon,
    const void *savedMean,
    const void *savedInvVariance,
    const cudnnActivationDescriptor_t activationDesc,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes
);

cudnnStatus_t (*cudnn_conv_bw_data_func)(
	cudnnHandle_t handle,
    const void *alpha,
    const cudnnFilterDescriptor_t wDesc,
    const void *w,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
);

cudnnStatus_t (*cudnn_conv_bw_filter_func)(
	cudnnHandle_t handle,
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    const cudnnFilterDescriptor_t dwDesc,
    void *dw
);

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

extern "C" {
	void block(int it) {
		int idx = get_idx();
		assert (idx >= 0);
		volatile bool* status_ar = client_request_status[idx];
		while (!status_ar[it]);
	}

	bool stop() {
		int idx = get_idx();
		assert (idx >= 0);
		pthread_mutex_lock(mutexes[idx]);
		bool res = client_stop[idx];
		if (res)
			client_stop_ack[idx] = true;
		pthread_mutex_unlock(mutexes[idx]);
		return res;
	}
}


cudaError_t cudaMalloc(void** devPtr, size_t size) {

	int idx = get_idx();
	assert (idx >= 0);
	DEBUG_PRINT("[IDX %d] Caught cudaMalloc! allocate region of %ld bytes\n", idx, size);

	cudaError_t err = cudaSuccess;


	if (malloc_func == NULL) {
		*(void **)(&malloc_func) = dlsym (RTLD_NEXT, "cudaMalloc");
		assert (malloc_func != NULL);
	}

	if (idx < *num_total_clients) {

		//wait for all kernels or memory operations to finish
		DEBUG_PRINT("About to block!\n");
		block(idx,  mutexes, kqueues);
		DEBUG_PRINT("Exit block!\n");


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
		//CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		err = (*malloc_func)(devPtr, size);
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

	int idx = get_idx();
	assert (idx >= 0);
	DEBUG_PRINT("[IDX %d] Caught cudaFree! free up address of %p\n", idx, devPtr);

	cudaError_t err = cudaSuccess;

	if (free_func == NULL) {
		*(void **)(&free_func) = dlsym (RTLD_NEXT, "cudaFree");
		assert (free_func != NULL);
	}

	if (idx < *num_total_clients) {

		// wait for all kernels or memory operations to finish
		block(idx,  mutexes, kqueues);
		free_record new_free_record = {devPtr};

		union func_data new_func_data;
		new_func_data.frecord = new_free_record;
		func_record new_record = {FREE_RECORD, new_func_data};

		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

	}
	else {
		err = (*free_func)(devPtr);
		CHECK_CUDA_ERROR(err);
	}

	return err;

}



cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {

	int idx = get_idx();
	assert (idx >= 0);
	DEBUG_PRINT("[IDX: %d], Caught cudaMemcpy!\n", idx);

	cudaError_t err = cudaSuccess;

	if (memcpy_func == NULL) {
		*(void **)(&memcpy_func) = dlsym (RTLD_NEXT, "cudaMemcpy");
		assert (memcpy_func != NULL);
	}

	if (idx < *num_total_clients) {

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

		err = (*memcpy_func)(dst, src, count, kind);
		CHECK_CUDA_ERROR(err);
		cudaError_t err_all = cudaDeviceSynchronize();
		CHECK_CUDA_ERROR(err_all);

	}

	return err;

}


cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {

	int idx = get_idx();
	assert (idx >= 0);

	DEBUG_PRINT("[IDX: %d] Caught cudaMemcpyAsync! src is %p, dst is %p, size is %d, stream is %d\n", idx, src, dst, count, stream);

	if (memcpy_async_func == NULL) {
		*(void **)(&memcpy_async_func) = dlsym (RTLD_NEXT, "cudaMemcpyAsync");
		assert (memcpy_async_func != NULL);
	}

	cudaError_t err = cudaSuccess;

	if (idx < *num_total_clients) {

		//wait for all kernels or memory operations to finish
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
		//CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		err = (*memcpy_async_func)(dst, src, count, kind, stream); // TODO: not sure about which stream to use here
		CHECK_CUDA_ERROR(err);
	}

	return err;

}


cudaError_t cudaMemset(void* devPtr, int  value, size_t count ) {

	int idx = get_idx();
	assert (idx >= 0);

	DEBUG_PRINT("[IDX: %d] Caught cudaMemset!\n", idx);
	cudaError_t err = cudaSuccess;

	if (memset_func == NULL) {
		*(void **)(&memset_func) = dlsym (RTLD_NEXT, "cudaMemset");
		assert (memset_async_func != NULL);
	}

	if (idx < *num_total_clients) {

		block(idx,  mutexes, kqueues);

		memset_record new_memset_record = {devPtr, value, count, 0, false};

		union func_data new_func_data;
		new_func_data.msetrecord = new_memset_record;
		func_record new_record = {MEMSET_RECORD, new_func_data};

		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		// although async, wait for debugging purposes
		block(idx,  mutexes, kqueues);
	}
	else {

		err = (*memset_func)(devPtr, value, count);
		CHECK_CUDA_ERROR(err);

	}

	return err;
}


cudaError_t cudaMemsetAsync ( void* devPtr, int  value, size_t count, cudaStream_t stream) {

	int idx = get_idx();
	assert (idx >= 0);

	DEBUG_PRINT("[IDX: %d] Caught cudaMemsetAsync!\n", idx);

	if (memset_async_func == NULL) {
		*(void **)(&memset_async_func) = dlsym (RTLD_NEXT, "cudaMemsetAsync");
		assert (memset_async_func != NULL);
	}

	cudaError_t err = cudaSuccess;

	if (idx < *num_total_clients) {

		block(idx,  mutexes, kqueues);

		memset_record new_memset_record = {devPtr, value, count, stream, true};

		union func_data new_func_data;
		new_func_data.msetrecord = new_memset_record;
		func_record new_record = {MEMSET_RECORD, new_func_data};

		pthread_mutex_lock(mutexes[idx]);
		kqueues[idx]->push(new_record);
		pthread_mutex_unlock(mutexes[idx]);

		// although async, wait for debugging purposes
		block(idx,  mutexes, kqueues);

	}
	else {
		cudaError_t err = (*memset_async_func)(devPtr, value, count, stream);
		CHECK_CUDA_ERROR(err);
	}

	return err;

}


cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {


	int idx = get_idx();
	assert (idx >= 0);

	if (kernel_func == NULL) {
		*(void **)(&kernel_func) = dlsym (RTLD_NEXT, "cudaLaunchKernel");
		assert (kernel_func != NULL);
	}

	cudaError_t err = cudaSuccess;
	kernel_record new_kernel_record;
	bool wait = false;

	if (idx < *num_total_clients) {

		pthread_mutex_lock(mutexes[idx]);


		new_kernel_record = {func, gridDim, blockDim, args, sharedMem, stream, false, 0};
		union func_data new_func_data;
		new_func_data.krecord = new_kernel_record;
		func_record new_record = {KERNEL_RECORD, new_func_data};
		kqueues[idx]->push(new_record);
		func_indexes[idx] += 1;

		pthread_mutex_unlock(mutexes[idx]);
		block(idx,  mutexes, kqueues);

	}
	else {
		DEBUG_PRINT("[INTERCEPTER] about to submit %p\n", func);

		err = (*kernel_func)(func, gridDim, blockDim, args, sharedMem, stream);
		CHECK_CUDA_ERROR(err); // this checks kernel-launching errors
		DEBUG_PRINT("SUBMITTED\n");

	}
	return err;
}
