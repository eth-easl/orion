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
volatile pid_t thread_ids[5]; // 2*N threads + scheduler

queue<func_record>* kqueues[2] = {&kqueue0, &kqueue1};
pthread_mutex_t* mutexes[2] = {&mutex0, &mutex1};
vector<char*>* func_names[2] = {&fnames0, &fnames1};
char* model_names[2];

int func_indexes[2] = {0, 0};

cudaStream_t client_streams[2];
bool streams_set[2] = {false, false};

using namespace boost::interprocess;

// new
volatile int* shmem = NULL;
volatile int* streams_shmem = NULL;
mapped_region* region;
mapped_region* streams_region;
cudaStream_t lp_stream;
cudaStream_t hp_stream;

cudaEvent_t lp_event;
cudaEvent_t hp_event;

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

void init() {

	pid_t pid = getpid();

	// map shared memory regions to talk with the scheduler

	// for requests
	std::string shmem_string_name = "client" + std::to_string(pid);
    const char* shmem_name = shmem_string_name.c_str();
	shared_memory_object shm (open_only, shmem_name, read_write);
    region = new mapped_region(shm, read_write);
	shmem = (int*)(region->get_address());

	// for streams
	std::string streams_shmem_string_name = "client_streams" + std::to_string(pid);
    const char* streams_shmem_name = streams_shmem_string_name.c_str();
	shared_memory_object streams_shm(open_only, streams_shmem_name, read_write);
    streams_region = new mapped_region(streams_shm, read_write);
	streams_shmem = (int*)(streams_region->get_address());

	// create streams
	int* lp = (int*)malloc(sizeof(int));
	int* hp = (int*)malloc(sizeof(int));

	CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(lp, hp));
	cudaStreamCreateWithPriority(&lp_stream, cudaStreamNonBlocking, 0);
	cudaStreamCreateWithPriority(&hp_stream, cudaStreamNonBlocking, *hp);

	printf("Highest stream priority is %d, lowest stream priority is %d\n", *hp, *lp);
	printf("LP stream: %d, hp stream: %d\n", lp_stream, hp_stream);

	CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&lp_event, cudaEventDisableTiming));
	CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&hp_event, cudaEventDisableTiming));

}

std::pair<cudaStream_t, cudaEvent_t> push_and_wait(int value, bool wait_for_stream) {

	*shmem = value;
	while (*shmem == value);
	if (wait_for_stream) {
		while (*streams_shmem == -1);
		cudaStream_t sched_stream = (*streams_shmem == 0) ? lp_stream : hp_stream;
		cudaEvent_t sched_event = (*streams_shmem == 0) ? lp_event : hp_event;
		*streams_shmem = -1;
		std::pair<cudaStream_t, cudaEvent_t> sched_pair(sched_stream, sched_event);
		return sched_pair;
	}
	else {
		std::pair<cudaStream_t, cudaEvent_t> sched_pair(lp_stream, lp_event);
		return sched_pair;
	}
}

cudaError_t cudaMalloc(void** devPtr, size_t size) {

	if (shmem==NULL) {
		init();
	}

	push_and_wait(MALLOC_RECORD, false);

	int idx = get_idx();
	assert (idx >= 0);
	DEBUG_PRINT("[IDX %d] Caught cudaMalloc! allocate region of %ld bytes\n", idx, size);

	cudaError_t err = cudaSuccess;

	if (malloc_func == NULL) {
		*(void **)(&malloc_func) = dlsym (RTLD_NEXT, "cudaMalloc");
		assert (malloc_func != NULL);
	}

	//CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	err = (*malloc_func)(devPtr, size);
	CHECK_CUDA_ERROR(err);
	cudaError_t err_all = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(err_all);
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

	push_and_wait(FREE_RECORD, false);

	err = (*free_func)(devPtr);
	CHECK_CUDA_ERROR(err);

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

	err = (*memcpy_func)(dst, src, count, kind);
	CHECK_CUDA_ERROR(err);
	cudaError_t err_all = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(err_all);

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
	std::pair<cudaStream_t, cudaEvent_t> sched_pair = push_and_wait(MEMCPY_RECORD, true);

	err = (*memcpy_async_func)(dst, src, count, kind, sched_pair.first); // TODO: not sure about which stream to use here
	CHECK_CUDA_ERROR(err);
	CHECK_CUDA_ERROR(cudaEventRecord(sched_pair.second, sched_pair.first));
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

	err = (*memset_func)(devPtr, value, count);
	CHECK_CUDA_ERROR(err);

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

	std::pair<cudaStream_t, cudaEvent_t> sched_pair = push_and_wait(MEMSET_RECORD, true);

	cudaError_t err = cudaSuccess;
	err = (*memset_async_func)(devPtr, value, count, sched_pair.first);
	CHECK_CUDA_ERROR(err);
	CHECK_CUDA_ERROR(cudaEventRecord(sched_pair.second, sched_pair.first));
	return err;

}


cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {


	int idx = get_idx();
	assert (idx >= 0);

	// TODO: remove this
	// if (idx < 2)
	// 	block(idx,  mutexes, kqueues);

	//if (idx < 2)
	//	DEBUG_PRINT("------------------------- IDX %d, model name is %s\n", idx, model_names[idx]);

	//DEBUG_PRINT("[INTERCEPTER-CATCH-%d] Captured a cudaLaunchKernel! function ptr is %p, stream is %d, gridDim is %d, blockDim is %d, sharedMem is %ld\n", idx, func, stream, gridDim, blockDim, sharedMem);
	//print_kernel_invocation(func_indexes[idx], gridDim, blockDim);

	if (kernel_func == NULL) {
		*(void **)(&kernel_func) = dlsym (RTLD_NEXT, "cudaLaunchKernel");
		assert (kernel_func != NULL);
	}

	std::pair<cudaStream_t, cudaEvent_t> sched_pair = push_and_wait(KERNEL_RECORD, true);

	cudaError_t err = cudaSuccess;
	kernel_record new_kernel_record;
	bool wait = false;

	DEBUG_PRINT("[INTERCEPTER] about to submit %p\n", func);

	err = (*kernel_func)(func, gridDim, blockDim, args, sharedMem, sched_pair.first);
	CHECK_CUDA_ERROR(err); // this checks kernel-launching errors
	CHECK_CUDA_ERROR(cudaEventRecord(sched_pair.second, sched_pair.first));
	return err;
}