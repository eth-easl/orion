#include "utils.h"

cudaError_t (*kernel_function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
cudaError_t (*memcpy_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t (*memcpy_async_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t (*malloc_function)(void** devPtr, size_t size);
cudnnStatus_t (*cudnn_conv_function)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) ;
cudnnStatus_t (*cudnn_bnorm_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);
cudnnStatus_t (*cudnn_bnorm_infer_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon);
cudnnStatus_t (*cudnn_rnn_function)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes);
cublasStatus_t (*cublas_sgemm_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
cublasStatus_t (*cublas_sgemm_strided_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount);

void pop_from_queue(queue<struct func_record>* &client_queue, pthread_mutex_t* &client_mutex) {
	pthread_mutex_lock(client_mutex);
	client_queue->pop();
	pthread_mutex_unlock(client_mutex);
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

	// CUBLAS sgemm
	*(void **)(&cublas_sgemm_function) = dlsym(RTLD_DEFAULT, "cublasSgemm_v2");
	assert(cublas_sgemm_function != NULL);


	// CUBLAS sgemm strided
	*(void **)(&cublas_sgemm_strided_function) = dlsym(RTLD_DEFAULT, "cublasSgemmStridedBatched");
	assert(&cublas_sgemm_strided_function != NULL);

}


void schedule_kernel(struct func_record frecord, cudaStream_t sched_stream) {

    // case 1
	if (frecord.type == KERNEL_RECORD) {
		DEBUG_PRINT("found a new kernel record! kernel func is %p\n", kernel_function);
		kernel_record record = frecord.data.krecord;
		(*kernel_function)(record.func, record.gridDim, record.blockDim, record.args, record.sharedMem, sched_stream);
	}

	else if (frecord.type == MEMCPY_RECORD) {
		DEBUG_PRINT("found a new memcpy record!\n");
		memcpy_record record = frecord.data.mrecord;
		if (not record.async) {
			(*memcpy_function)(record.dst, record.src, record.count, record.kind);
		} else {
			(*memcpy_async_function)(record.dst, record.src, record.count, record.kind, sched_stream);
		}

	}

	else if (frecord.type == MALLOC_RECORD) {
		DEBUG_PRINT("found a new malloc record!\n");
		malloc_record record = frecord.data.malrecord;
		(*malloc_function)(record.devPtr, record.size);
    }

	else if (frecord.type == CUDNN_CONV_RECORD) {
	    DEBUG_PRINT("found a new cudnn conv record!\n");
		cudnnConvolutionForward_record record = frecord.data.cudnnConvRecord;
		cudnnSetStream(record.handle, 0);
		(*cudnn_conv_function)(record.handle, record.alpha, record.xDesc, record.x, record.wDesc, record.w, record.convDesc, record.algo, record.workSpace, record.workSpaceSizeInBytes, record.beta, record.yDesc, record.y);
		cudnnSetStream(record.handle, 0); // TODO: I want to set the default stream here
	}

	else if (frecord.type == CUDNN_BNORM_RECORD) {
		DEBUG_PRINT("found a new bnorm record!\n");
		cudnnBatchNormalizationForwardTrainingEx_record record = frecord.data.cudnnBNormRecord;
		cudnnSetStream(record.handle, 0);
		(*cudnn_bnorm_function)(record.handle, record.mode, record.bnOps, record.alpha, record.beta, record.xDesc, record.xData, record.zDesc, record.zData, record.yDesc, record.yData, record.bnScaleBiasMeanVarDesc, record.bnScaleData, record.bnBiasData, record.exponentialAverageFactor, record.resultRunningMeanData, record.resultRunningVarianceData, record.epsilon, record.saveMean, record.saveInvVariance, record.activationDesc, record.workspace, record.workSpaceSizeInBytes, record.reserveSpace, record.reserveSpaceSizeInBytes);
		cudnnSetStream(record.handle, 0); // TODO: I want to set the default stream here
	}

	else if (frecord.type == CUDNN_BNORM_INF_RECORD) {
		DEBUG_PRINT("found a new bnorm inf record!\n");
		cudnnBatchNormalizationForwardInference_record record = frecord.data.cudnnBNormInfRecord;
		cudnnSetStream(record.handle, 0);
		(*cudnn_bnorm_infer_function)(record.handle, record.mode, record.alpha, record.beta, record.xDesc, record.x, record.yDesc, record.y, record.bnScaleBiasMeanVarDesc, record.bnScale, record.bnBias, record.estimatedMean, record.estimatedVariance, record.epsilon);
		cudnnSetStream(record.handle, 0);
	}

	else if (frecord.type == CUDNN_RNN_INF_RECORD) {
		DEBUG_PRINT("found a new cudnn rnn inf record!\n");
		cudnnRNNForwardInference_record record = frecord.data.cudnnRnnInfRecord;
		(*cudnn_rnn_function)(record.handle, record.rnnDesc, record.seqLength, record.xDesc, record.x, record.hxDesc, record.hx, record.cxDesc, record.cx, record.wDesc, record.w, record.yDesc, record.y, record.hyDesc, record.hy, record.cyDesc, record.cy, record.workspace, record.workSpaceSizeInBytes);
	}

    else if (frecord.type == CUBLAS_SGEMM_RECORD) {
		DEBUG_PRINT("found a new sgemm record!\n");

		// TODO: what to do about streams?
		cublasSgemm_record record = frecord.data.cublasSgemmRecord;
		DEBUG_PRINT("handle is %p\n", record.handle);
		(*cublas_sgemm_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.B, record.ldb, record.beta, record.C, record.ldc);
	}

	else if (frecord.type == CUBLAS_SGEMM_STRIDED_RECORD) {
		DEBUG_PRINT("found a new sgemm strided record!\n");

		cublasSgemmStridedBatched_record record = frecord.data.cublasSgemmStridedRecord;
		DEBUG_PRINT("handle is %p\n", record.handle);
		(*cublas_sgemm_strided_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.strideA, record.B, record.ldb, record.strideB, record.beta, record.C, record.ldc, record.strideC, record.batchCount);
    }

}


void schedule_pair(vector<func_record*> &frecords, queue<struct func_record>** &buffers, pthread_mutex_t** &mutexes, vector<vector<op_info>> &op_info_vector, int idx0, int idx1, int max_sms, cudaStream_t stream) {

	op_info op_info_0 = op_info_vector[0][idx0];
	op_info op_info_1 = op_info_vector[1][idx1];

	if (op_info_0.profile > -1 && (op_info_0.profile == op_info_1.profile)) {
		schedule_kernel(*(frecords[0]), stream);
		pop_from_queue(buffers[0], mutexes[0]);
	}
	// different profiles
	else if (op_info_0.sm_used < max_sms && op_info_1.sm_used < max_sms) {
		schedule_kernel(*(frecords[0]), stream); // TODO: fix streams + priority
		schedule_kernel(*(frecords[1]), stream); // TODO: fix streams + priority
		pop_from_queue(buffers[0], mutexes[0]);
		pop_from_queue(buffers[0], mutexes[0]);
	}

	else if (op_info_0.sm_used >= max_sms && op_info_1.sm_used < max_sms) {
		schedule_kernel(*(frecords[0]), stream); // TODO: fix streams + priority
		schedule_kernel(*(frecords[1]), stream); // TODO: fix streams + priority
		pop_from_queue(buffers[0], mutexes[0]);
		pop_from_queue(buffers[0], mutexes[0]);
	}

	else if (op_info_0.sm_used < max_sms && op_info_1.sm_used >= max_sms) {
		schedule_kernel(*(frecords[0]), stream); // TODO: fix streams + priority
		schedule_kernel(*(frecords[1]), stream); // TODO: fix streams + priority
		pop_from_queue(buffers[0], mutexes[0]);
		pop_from_queue(buffers[0], mutexes[0]);
	}

	else {
		schedule_kernel(*(frecords[0]), stream);
		pop_from_queue(buffers[0], mutexes[0]);
	}
}