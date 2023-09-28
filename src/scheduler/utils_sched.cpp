#include "utils_sched.h"

cudaError_t (*kernel_function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
cudaError_t (*memcpy_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t (*memcpy_async_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t (*malloc_function)(void** devPtr, size_t size);
cudaError_t (*free_function)(void* devPtr);
cudaError_t (*memset_function)(void* devPtr, int  value, size_t count);
cudaError_t (*memset_async_function)(void* devPtr, int  value, size_t count, cudaStream_t stream);

cudnnStatus_t (*cudnn_create_function)(cudnnHandle_t *handle);
cudnnStatus_t (*cudnn_bnorm_reserve_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes);
cudnnStatus_t (*cudnn_conv_function)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) ;
cudnnStatus_t (*cudnn_bnorm_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);
cudnnStatus_t (*cudnn_bnorm_infer_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon);
cudnnStatus_t (*cudnn_rnn_function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes);
cudnnStatus_t (*cudnn_rnn_train_function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);

cudnnStatus_t (*cudnn_bnorm_bw_function)(
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

cudnnStatus_t (*cudnn_conv_bw_data_function)(
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

cudnnStatus_t (*cudnn_conv_bw_filter_function)(
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

cublasStatus_t (*cublas_sgemm_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
cublasStatus_t (*cublas_sgemm_strided_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount);

extern cudnnHandle_t* global_handle0;
extern cudnnHandle_t* global_handle1;

extern int status;
extern int* seen;
extern int* num_client_kernels;

void process_eval(vector<vector<float>> &client_durations) {

	int num_clients = client_durations.size();
	for (int i=0; i<num_clients; i++) {
		vector<float> client_stats = client_durations[i];
		// remove first two iters
		client_stats.erase(client_stats.begin());
		client_stats.erase(client_stats.begin());

		sort(client_stats.begin(), client_stats.end());
		int client_len = client_stats.size();
		float p50 = client_stats[client_len/2];
		float p95 = client_stats[(client_len*95/100)];
		float p99 = client_stats[(client_len*99/100)];
		printf("Client %d, p50=%f, p95=%f, p99=%f\n", i, p50, p95, p99);
	}
}


void pop_from_queue(queue<struct func_record>* client_queue, pthread_mutex_t* client_mutex, int idx) {
	pthread_mutex_lock(client_mutex);
	client_queue->pop();
	pthread_mutex_unlock(client_mutex);
}

void create_streams(cudaStream_t** sched_streams, int num, bool reef) {

	int* lp = (int*)malloc(sizeof(int));
	int* hp = (int*)malloc(sizeof(int));

	CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(lp, hp));

	DEBUG_PRINT("Highest stream priority is %d, lowest stream priority is %d\n", *hp, *lp);
	assert(*lp==0);

	for (int i=0; i<num-1; i++) {
		sched_streams[i] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
		cudaStreamCreateWithPriority(sched_streams[i], cudaStreamNonBlocking, 0);
	}

	// client num-1 is high priority
	if (!reef) {
		sched_streams[num-1] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
		cudaStreamCreateWithPriority(sched_streams[num-1], cudaStreamNonBlocking, *hp);
	}
	else {
		sched_streams[num-1] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
		cudaStreamCreateWithPriority(sched_streams[num-1], cudaStreamNonBlocking, 0);
	}

}

void create_events(cudaEvent_t*** events, int num) {

	// per-stream event
	for (int i=0; i<num; i++) {
		events[i] = (cudaEvent_t**)malloc(30000*sizeof(cudaEvent_t*));
		for (int j=0; j<30000; j++) {
			//printf("create %d, %d\n", i, j);
			events[i][j] = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
			CHECK_CUDA_ERROR(cudaEventCreateWithFlags(events[i][j], cudaEventDisableTiming));
		}
	}
}

void register_functions() {

    // for kernel
	*(void **)(&kernel_function) = dlsym(RTLD_DEFAULT, "cudaLaunchKernel");
	assert(kernel_function != NULL);

	// for memcpy
	*(void **)(&memcpy_function) = dlsym (RTLD_DEFAULT, "cudaMemcpy");
	assert(memcpy_function != NULL);

	// for memcpy_async
	*(void **)(&memcpy_async_function) = dlsym (RTLD_DEFAULT, "cudaMemcpyAsync");
	assert(memcpy_async_function != NULL);

	// for malloc
	*(void **)(&malloc_function) = dlsym (RTLD_DEFAULT, "cudaMalloc");
	assert(malloc_function != NULL);

	// for free
	*(void **)(&free_function) = dlsym (RTLD_DEFAULT, "cudaFree");
	assert(free_function != NULL);

	// for memset
	*(void **)(&memset_function) = dlsym (RTLD_DEFAULT, "cudaMemset");
	assert (memset_function != NULL);

	// for memset_async
	*(void **)(&memset_async_function) = dlsym (RTLD_DEFAULT, "cudaMemsetAsync");
	assert (memset_async_function != NULL);

	// for cudnn create
	*(void **)(&cudnn_create_function) = dlsym(RTLD_DEFAULT, "cudnnCreate");
	assert(cudnn_create_function != NULL);

	// for cudnn_bnorm_reserve
	*(void **)(&cudnn_bnorm_reserve_function) = dlsym(RTLD_DEFAULT, "cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
	assert(cudnn_bnorm_reserve_function != NULL);

	// for cudnn conv
	*(void **)(&cudnn_conv_function) = dlsym(RTLD_DEFAULT, "cudnnConvolutionForward");
	assert(cudnn_conv_function != NULL);

	// for bnorm train
	*(void **)(&cudnn_bnorm_function) = dlsym(RTLD_DEFAULT, "cudnnBatchNormalizationForwardTrainingEx");
	assert(cudnn_bnorm_function != NULL);

	// for bnorm infer
	*(void **)(&cudnn_bnorm_infer_function) = dlsym(RTLD_DEFAULT, "cudnnBatchNormalizationForwardInference");
	assert(cudnn_bnorm_infer_function != NULL);

	// for rnn infer
	*(void **)(&cudnn_rnn_function) = dlsym(RTLD_DEFAULT, "cudnnRNNForwardInference");
	assert(cudnn_rnn_function != NULL);

	// for rnn train
	*(void **)(&cudnn_rnn_train_function) = dlsym(RTLD_DEFAULT, "cudnnRNNForwardTraining");
	assert(cudnn_rnn_train_function != NULL);

	// for bnorm backward
	*(void **)(&cudnn_bnorm_bw_function) = dlsym(RTLD_DEFAULT, "cudnnBatchNormalizationBackwardEx");
	assert(cudnn_bnorm_bw_function != NULL);

	// for conv data backward
	*(void **)(&cudnn_conv_bw_data_function) = dlsym(RTLD_DEFAULT, "cudnnConvolutionBackwardData");
	assert(cudnn_conv_bw_data_function != NULL);

	// for conv filter backward
	*(void **)(&cudnn_conv_bw_filter_function) = dlsym(RTLD_DEFAULT, "cudnnConvolutionBackwardFilter");
	assert(cudnn_conv_bw_filter_function != NULL);

	// CUBLAS sgemm
	*(void **)(&cublas_sgemm_function) = dlsym(RTLD_DEFAULT, "cublasSgemm_v2");
	assert(cublas_sgemm_function != NULL);

	// CUBLAS sgemm strided
	*(void **)(&cublas_sgemm_strided_function) = dlsym(RTLD_DEFAULT, "cublasSgemmStridedBatched");
	assert(&cublas_sgemm_strided_function != NULL);

}

void schedule_kernel(struct func_record frecord, cudaStream_t* sched_stream, int idx, cudaEvent_t* event, int* seen, int* event_ids, int evid) {

	switch (frecord.type) {
		case KERNEL_RECORD: {
			DEBUG_PRINT("found a new kernel record from idx %d! kernel func is %p\n", idx, kernel_function);
			kernel_record record = frecord.data.krecord;
			(*kernel_function)(record.func, record.gridDim, record.blockDim, record.args, record.sharedMem, *sched_stream);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case MEMCPY_RECORD: {
			memcpy_record record = frecord.data.mrecord;
			if (!record.async) {
				(*memcpy_function)(record.dst, record.src, record.count, record.kind);
			} else {
				(*memcpy_async_function)(record.dst, record.src, record.count, record.kind, *sched_stream);
			}
			break;
		}
		case MALLOC_RECORD: {
			DEBUG_PRINT("found a new malloc record from idx %d!\n", idx);
			malloc_record record = frecord.data.malrecord;
			(*malloc_function)(record.devPtr, record.size);
			break;
		}
		case FREE_RECORD: {
			DEBUG_PRINT("found a new FREE record from idx %d!\n", idx);
			free_record record = frecord.data.frecord;
			//(*free_function)(record.devPtr);
			break;
		}
		case MEMSET_RECORD: {
			memset_record record = frecord.data.msetrecord;
			if (record.async) {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_async_function)(record.devPtr, record.value, record.count, *sched_stream);
			}
			else {
				DEBUG_PRINT("found a new MEMSET-ASYNC record from idx %d!\n", idx);
				(*memset_function)(record.devPtr, record.value, record.count);
			}
			break;
		}
		case CUDNN_CONV_RECORD: {

			DEBUG_PRINT("found a new cudnn conv record from idx %d!\n", idx);
			cudnnConvolutionForward_record record = frecord.data.cudnnConvRecord;
			cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
			status = cudnnSetStream(record.handle, *sched_stream);
			assert (status == CUDNN_STATUS_SUCCESS);
			(*cudnn_conv_function)(record.handle, record.alpha, record.xDesc, record.x, record.wDesc, record.w, record.convDesc, record.algo, record.workSpace, record.workSpaceSizeInBytes, record.beta, record.yDesc, record.y);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_BNORM_RECORD: {
			//DEBUG_PRINT("found a new bnorm training record from idx %d!\n", idx);
			cudnnBatchNormalizationForwardTrainingEx_record record = frecord.data.cudnnBNormRecord;

			//printf("Got a CUDNN operation from client %d, handle is %p\n", idx, record.handle);
			cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

			//printf("Stream is %d\n", *sched_stream);
			status = cudnnSetStream(record.handle, *sched_stream);
			assert (status == CUDNN_STATUS_SUCCESS);
			(*cudnn_bnorm_function)(
				record.handle,
				record.mode,
				record.bnOps,
				record.alpha,
				record.beta,
				record.xDesc,
				record.xData,
				record.zDesc,
				record.zData,
				record.yDesc,
				record.yData,
				record.bnScaleBiasMeanVarDesc,
				record.bnScaleData,
				record.bnBiasData,
				record.exponentialAverageFactor,
				record.resultRunningMeanData,
				record.resultRunningVarianceData,
				record.epsilon,
				record.saveMean,
				record.saveInvVariance,
				record.activationDesc,
				record.workspace,
				record.workSpaceSizeInBytes,
				record.reserveSpace,
				record.reserveSpaceSizeInBytes
			);
			//cudnnSetStream(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_BNORM_INF_RECORD: {
			DEBUG_PRINT("found a new bnorm inf record from idx %d!\n", idx);
			cudnnBatchNormalizationForwardInference_record record = frecord.data.cudnnBNormInfRecord;
			//printf("Got a CUDNN operation from client %d, handle is %p\n", idx, record.handle);
			cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
			status = cudnnSetStream(record.handle, *sched_stream);
			assert (status == CUDNN_STATUS_SUCCESS);
			(*cudnn_bnorm_infer_function)(record.handle, record.mode, record.alpha, record.beta, record.xDesc, record.x, record.yDesc, record.y, record.bnScaleBiasMeanVarDesc, record.bnScale, record.bnBias, record.estimatedMean, record.estimatedVariance, record.epsilon);
			//cudnnSetStream(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_RNN_INF_RECORD: {
			DEBUG_PRINT("found a new cudnn rnn inf record from idx %d!\n", idx);
			cudnnRNNForwardInf_record record = frecord.data.cudnnRnnInfRecord;
			cudnnSetStream(record.handle, *sched_stream);
			(*cudnn_rnn_function)(record.handle, record.rnnDesc, record.seqLength, record.xDesc, record.x, record.hxDesc, record.hx, record.cxDesc, record.cx, record.wDesc, record.w, record.yDesc, record.y, record.hyDesc, record.hy, record.cyDesc, record.cy, record.workspace, record.workSpaceSizeInBytes);
			//cudnnSetStream(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_RNN_TRAIN_RECORD: {
			DEBUG_PRINT("found a new cudnn rnn train record from idx %d!\n", idx);
			cudnnRNNForwardTraining_record record = frecord.data.cudnnRnnTrainRecord;
			cudnnSetStream(record.handle, *sched_stream);
			(*cudnn_rnn_train_function)(
				record.handle,
				record.rnnDesc,
				record.seqLength,
				record.xDesc,
				record.x,
				record.hxDesc,
				record.hx,
				record.cxDesc,
				record.cx,
				record.wDesc,
				record.w,
				record.yDesc,
				record.y,
				record.hyDesc,
				record.hy,
				record.cyDesc,
				record.cy,
				record.workspace,
				record.workSpaceSizeInBytes,
				record.reserveSpace,
				record.reserveSpaceSizeInBytes
			);
			//cudnnSetStream(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_BNORM_BACKWARD_RECORD: {
			DEBUG_PRINT("found a new cudnn batch norm backw record from idx %d!\n", idx);

			cudnnBatchNormalizationBackwardEx_record record = frecord.data.cudnnBNormBackRecord;
			cudnnSetStream(record.handle, *sched_stream);
			(*cudnn_bnorm_bw_function)(
					record.handle,
					record.mode,
					record.bnOps,
					record.alphaDataDiff,
					record.betaDataDiff,
					record.alphaParamDiff,
					record.betaParamDiff,
					record.xDesc,
					record.xData,
					record.yDesc,
					record.yData,
					record.dyDesc,
					record.dyData,
					record.dzDesc,
					record.dzData,
					record.dxDesc,
					record.dxData,
					record.dBnScaleBiasDesc,
					record.bnScaleData,
					record.bnBiasData,
					record.dBnScaleData,
					record.dBnBiasData,
					record.epsilon,
					record.savedMean,
					record.savedInvVariance,
					record.activationDesc,
					record.workspace,
					record.workSpaceSizeInBytes,
					record.reserveSpace,
					record.reserveSpaceSizeInBytes
			);
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_CONV_DATA_RECORD: {
			DEBUG_PRINT("found a new cudnn conv data backw record from idx %d!\n", idx);

			cudnnConvolutionBackwardData_record record = frecord.data.cudnnConvBackDataRecord;
			cudnnSetStream(record.handle, *sched_stream);
			DEBUG_PRINT("submit!\n");
			(*cudnn_conv_bw_data_function)(
					record.handle,
					record.alpha,
					record.wDesc,
					record.w,
					record.dyDesc,
					record.dy,
					record.convDesc,
					record.algo,
					record.workSpace,
					record.workSpaceSizeInBytes,
					record.beta,
					record.dxDesc,
					record.dx
			);

			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUDNN_CONV_FILTER_RECORD: {
			DEBUG_PRINT("found a new cudnn conv filter backw record from idx %d!\n", idx);

			cudnnConvolutionBackwardFilter_record record = frecord.data.cudnnConvBackFilterRecord;
			cudnnSetStream(record.handle, *sched_stream);
			(*cudnn_conv_bw_filter_function)(
					record.handle,
					record.alpha,
					record.xDesc,
					record.x,
					record.dyDesc,
					record.dy,
					record.convDesc,
					record.algo,
					record.workSpace,
					record.workSpaceSizeInBytes,
					record.beta,
					record.dwDesc,
					record.dw
			);

			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUBLAS_SGEMM_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm record from idx %d!\n", idx);

			cublasSgemm_record record = frecord.data.cublasSgemmRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			(*cublas_sgemm_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.B, record.ldb, record.beta, record.C, record.ldc);
			//cublasSetStream_v2(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		case CUBLAS_SGEMM_STRIDED_RECORD: {
			DEBUG_PRINT("found a new cublas sgemm strided record from idx %d!\n", idx);

			cublasSgemmStridedBatched_record record = frecord.data.cublasSgemmStridedRecord;
			cublasSetStream_v2(record.handle, *sched_stream);
			(*cublas_sgemm_strided_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.strideA, record.B, record.ldb, record.strideB, record.beta, record.C, record.ldc, record.strideC, record.batchCount);
			//cublasSetStream_v2(record.handle, 0);
			// CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
			event_ids[evid] += 1;
			seen[idx] += 1;
			break;
		}
		default:
			printf("UNSUPPORTED OPERATION - ABORT\n");
			abort();

	}
	DEBUG_PRINT("Return from schedule, seen[%d] is %d!\n", idx, seen[idx]);
	CHECK_CUDA_ERROR(cudaEventRecord(*event, *sched_stream));
	//event_ids[evid] += 1;
}